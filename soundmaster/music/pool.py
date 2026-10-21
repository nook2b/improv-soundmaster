"""Пул записанных мелодий: по нескольку на жанр, каждая — один инструмент.

Зачем: ручные кнопки жанров на пульте (повторное нажатие — следующая мелодия) и резерв, когда
генератор недоступен (нет интернета). Записывает их `python -m soundmaster make-music-pool`:
по умолчанию ElevenLabs Music v2.5 (выиграл слепое сравнение 2026-10-21), `--engine lyria` — Lyria
RealTime по тем же пресетам словаря, что и живая музыка.

Каждая мелодия — бесшовная петля: хвост записи плавно наложен на начало, в полосе 2–5 кГц убрано
3 дБ (место для речи), громкость выровнена.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..registry import PoolTrack, Registry

SR = 48000
LOOP_XFADE = 4.0  # с: столько хвоста накладывается на начало петли
SKIP_START = 2.0  # с: начало записи Lyria (разгон) не берём
TARGET_RMS_DB = -24.0
SPEECH_EQ = "equalizer=f=3200:t=o:w=1.3:g=-3"  # вырез 2–5 кГц: согласные речи остаются разборчивыми

ELEVEN_URL = "https://api.elevenlabs.io/v1/music"
ELEVEN_MODEL = "music_v2_5"  # по умолчанию API берёт старую music_v1
ELEVEN_SKIP = 0.5  # с: начало записи ElevenLabs не берём (атака первой ноты)
ELEVEN_TEMPO = {"slow": "slow tempo", "medium": "relaxed moderate pace", "fast": "gently moving pace"}
# Чего не должно быть в мелодии пула: «используйте negative_styles щедро» (гайд ElevenLabs).
ELEVEN_NEGATIVE = ["vocals", "humming", "choir", "drums", "percussion", "bassline", "chords", "accompaniment",
                   "pads", "drones", "orchestra", "string section", "second instrument", "build-up", "crescendo",
                   "climax", "loud accents", "fast tempo", "busy melody", "sound effects"]


def make_loop(x: np.ndarray, seconds: float, sr: int = SR, xfade: float = LOOP_XFADE) -> np.ndarray:
    """Петля длиной seconds: хвост (xfade с после конца) плавно накладывается на начало."""
    n, f = int(seconds * sr), int(min(xfade, seconds / 2) * sr)
    if len(x) < n + f:
        raise ValueError(f"мало звука для петли: {len(x) / sr:.1f} с, нужно {(n + f) / sr:.1f} с")
    out = x[:n].copy()
    k = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
    # Равная мощность: sqrt-кривые, чтобы на стыке не было провала громкости.
    out[:f] = x[:f] * np.sqrt(k) + x[n:n + f] * np.sqrt(1.0 - k)
    return out


def normalize(x: np.ndarray, rms_db: float = TARGET_RMS_DB, peak_db: float = -1.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    g = 10 ** (rms_db / 20) / rms
    peak = float(np.abs(x).max()) * g + 1e-9
    g *= min(1.0, 10 ** (peak_db / 20) / peak)
    return (x * g).astype(np.float32)


def busyness(x: np.ndarray, sr: int = SR) -> float:
    """Сколько нот в секунду (атаки по спектральному потоку): меньше — спокойнее, лучше под речь."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    n, hop = 2048, 512
    frames = (len(m) - n) // hop
    if frames < 8:
        return 0.0
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.log1p(np.abs(np.fft.rfft(m[idx] * np.hanning(n), axis=1)) * 10)
    flux = np.maximum(spec[1:] - spec[:-1], 0).sum(axis=1)
    w = max(1, int(sr / hop))  # окно ~1 с для порога
    pad = np.pad(flux, w, mode="edge")
    local = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)[: len(flux)]
    thr = local.mean(axis=1) + 1.0 * local.std(axis=1)
    peak = (flux > thr) & (flux >= np.roll(flux, 1)) & (flux >= np.roll(flux, -1))
    # не чаще одной атаки за 50 мс
    hits, last = 0, -10**9
    for i in np.nonzero(peak)[0]:
        if i - last >= int(0.05 * sr / hop):
            hits, last = hits + 1, i
    return hits / (len(m) / sr)


def write_flac(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.flac")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                    "-af", SPEECH_EQ, "-c:a", "flac", "-sample_fmt", "s16", str(tmp)],
                   input=np.ascontiguousarray(x, dtype="<f4").tobytes(), check=True, capture_output=True)
    tmp.replace(path)


async def record_lyria(preset: dict, seconds: float, api_key: str, timeout: float = 120.0) -> np.ndarray:
    """Записать seconds секунд музыки Lyria по пресету (prompts + config)."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    need = int(seconds * SR) * 4
    buf = bytearray()
    t0 = time.monotonic()
    async with client.aio.live.music.connect(model="models/lyria-realtime-exp") as s:
        await s.set_weighted_prompts(prompts=[types.WeightedPrompt(text=p["text"], weight=float(p["weight"]))
                                              for p in preset["prompts"]])
        if preset.get("config"):
            await s.set_music_generation_config(config=types.LiveMusicGenerationConfig(**preset["config"]))
        await s.play()
        async for msg in s.receive():
            for ch in (getattr(getattr(msg, "server_content", None), "audio_chunks", None) or []):
                buf += ch.data
            if len(buf) >= need:
                break
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"Lyria за {timeout:.0f} с дала {len(buf) / SR / 4:.1f} с звука")
    return np.frombuffer(bytes(buf[:need]), "<i2").astype(np.float32).reshape(-1, 2) / 32768.0


def _styles(text: str) -> list[str]:
    """Фразы описания без запретов («no …»): запреты у ElevenLabs — отдельным списком."""
    parts = (p.strip() for p in text.replace(";", ",").split(","))
    return [p for p in parts if p and not p.lower().startswith(("no ", "not "))]


def eleven_plan(registry: Registry, t: PoolTrack, seconds: float) -> dict:
    """План композиции ElevenLabs для мелодии пула: один блок, что звучит и чего не должно быть."""
    vocab = registry.vocabulary
    assert vocab is not None
    st = t.style
    g, m = vocab.genres[st.genre], vocab.moods[st.mood]
    bpm = vocab.preset(st)["config"].get("bpm", 66) + (10 if st.tempo == "medium" else 0)
    beat = g.config.get("mute_drums") is False  # жанр с битом (клуб): бочка разрешена
    pos = (_styles(vocab.instruments[st.instrument].text) + _styles(t.extra)
           + (_styles(g.solo) if g.solo else ["just one solo instrument playing alone"])
           + _styles(g.text) + _styles(m.text) + [ELEVEN_TEMPO.get(st.tempo, "slow tempo")]
           + [f"{bpm} BPM", "minimal underscore beneath spoken theatre dialogue",
              "sparse, lots of silence between phrases", "soft steady dynamics, constant intensity",
              "close-mic'd, dry, intimate room", "seamless loop-friendly repeating motif"])
    neg = [n for n in ELEVEN_NEGATIVE if not (beat and n in ("drums", "percussion"))
           and not (st.instrument == "doublebass" and n == "bassline")]
    if beat:
        neg += ["snare", "hi-hats", "claps"]
    return {"chunks": [{"text": "[Instrumental loop]", "duration_ms": int(seconds * 1000),
                        "positive_styles": pos[:50], "negative_styles": neg[:50], "context_adherence": "high"}]}


class ElevenFatal(RuntimeError):
    """Ошибка, которую повтор не исправит и которая помешает всем мелодиям (ключ, кредиты)."""


class PlanRejected(RuntimeError):
    """ElevenLabs отклонил план этой мелодии (например, слово похоже на название песни) — пропускаем её."""


def decode(data: bytes, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"],
                         input=data, capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<f4").reshape(-1, 2).copy()


async def record_eleven(plan: dict, key: str, timeout: float = 300.0) -> np.ndarray:
    """Записать мелодию ElevenLabs по плану. Ключ — ELEVENLABS_API_KEY (или подставляет прокси окружения)."""
    import aiohttp

    headers = {"xi-api-key": key} if key else {}
    async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=timeout)) as s:
        async with s.post(ELEVEN_URL, params={"output_format": "mp3_44100_192"}, headers=headers,
                          json={"composition_plan": plan, "model_id": ELEVEN_MODEL}) as r:
            body = await r.read()
            if r.status == 429 or r.status >= 500:
                raise RuntimeError(f"ElevenLabs {r.status}: {body[:200].decode(errors='replace')}")
            if r.status == 400 and b"bad_composition_plan" in body:
                raise PlanRejected(f"ElevenLabs отклонил описание: {body[:300].decode(errors='replace')}")
            if r.status != 200:
                raise ElevenFatal(f"ElevenLabs {r.status}: {body[:300].decode(errors='replace')}")
    return decode(body)


@dataclass
class PoolReport:
    made: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def select(registry: Registry, only: str | None = None) -> list[PoolTrack]:
    """Все мелодии пула или только жанр (`noir`) / мелодия (`noir/2`); через запятую — несколько."""
    tracks = [t for ts in registry.pool.values() for t in ts]
    if not only:
        return tracks
    out: list[PoolTrack] = []
    for part in (p.strip() for p in only.split(",") if p.strip()):
        genre, _, n = part.partition("/")
        if genre not in registry.pool:
            raise KeyError(f"в пуле нет жанра {genre!r}; есть: {', '.join(registry.pool)}")
        out += [t for t in registry.pool[genre] if (not n or str(t.n) == n) and t not in out]
    return out


def make_pool(registry: Registry, seconds: float = 60.0, parallel: int = 4, only: str | None = None,
              force: bool = False, progress: Callable[[str], None] = print,
              recorder: Callable | None = None, takes: int = 3, engine: str = "lyria") -> PoolReport:
    """Записать недостающие мелодии пула (force — перезаписать). recorder — для тестов.

    На каждую мелодию — takes дублей; остаётся самый спокойный (меньше нот в секунду): под речь
    актёров нужна редкая, неторопливая музыка. Сначала пишется «Нейтральный фон» (70% сцен —
    обычные современные), потом по первой мелодии каждого жанра, по второй и т. д. — если кредиты
    кончатся, у каждого жанра уже что-то будет.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("нужен ffmpeg (Mac: brew install ffmpeg)")
    if engine not in ("lyria", "elevenlabs"):
        raise RuntimeError(f"неизвестный генератор {engine!r}: lyria или elevenlabs")
    eleven = engine == "elevenlabs"
    key = os.environ.get("ELEVENLABS_API_KEY" if eleven else "GEMINI_API_KEY", "")
    if recorder is None and not key and not eleven:
        raise RuntimeError("нужен GEMINI_API_KEY (Lyria)")
    rec = recorder or (record_eleven if eleven else record_lyria)
    vocab = registry.vocabulary
    assert vocab is not None
    report = PoolReport()
    todo = []
    for t in select(registry, only):
        path = registry.sounds_dir / t.file
        if path.exists() and not force:
            report.skipped.append(t.file)
        else:
            todo.append((t, path))
    todo.sort(key=lambda tp: (tp[0].style.genre != "underscore", tp[0].n))
    total = len(todo)
    fatal: list[str] = []

    async def take(t: PoolTrack) -> tuple[np.ndarray, dict]:
        if eleven:
            plan = eleven_plan(registry, t, ELEVEN_SKIP + seconds + LOOP_XFADE)
            raw = await rec(plan, key)
            return make_loop(raw[int(ELEVEN_SKIP * SR):], seconds), {"model": ELEVEN_MODEL, "composition_plan": plan}
        preset = vocab.pool_preset(t.style, t.extra)
        preset["config"]["seed"] = random.randrange(1, 2**31)
        raw = await rec(preset, SKIP_START + seconds + LOOP_XFADE, key)
        return make_loop(raw[int(SKIP_START * SR):], seconds), {"model": "lyria-realtime-exp", "preset": preset}

    async def one(i: int, t: PoolTrack, path: Path, sem: asyncio.Semaphore) -> None:
        async with sem:
            name = f"{t.file} ({vocab.label(t.style)})"
            best: tuple[float, np.ndarray, dict] | None = None
            err = ""
            for _take in range(max(1, takes)):
                for attempt in (1, 2, 3):
                    if fatal:
                        err = fatal[0]
                        break
                    try:
                        x, meta = await take(t)
                        b = busyness(x)
                        if best is None or b < best[0]:
                            best = (b, x, meta)
                        break
                    except PlanRejected as e:  # повтор того же описания не поможет
                        err = str(e)
                        break
                    except ElevenFatal as e:  # ключ/кредиты: остальное тоже не запишется
                        err = str(e)
                        fatal.append(err)
                        break
                    except Exception as e:  # noqa: BLE001 — сеть/генератор: пробуем ещё, потом пропускаем
                        err = f"{type(e).__name__}: {e}"
                        if attempt < 3:
                            progress(f"повтор {attempt}: {name} — {err}")
                            await asyncio.sleep(5 * attempt)
            if best is None:
                report.errors.append((t.file, err))
                progress(f"[{len(report.made) + len(report.errors)}/{total}] ОШИБКА: {name} — {err}")
                return
            write_flac(path, normalize(best[1]))
            # Как получена мелодия (промпт, сид, модель) — чтобы удачную можно было повторить.
            path.with_suffix(".json").write_text(json.dumps(
                {**best[2], "busyness": round(best[0], 2), "made": time.strftime("%Y-%m-%d %H:%M")},
                ensure_ascii=False, indent=1),
                encoding="utf-8")
            report.made.append(t.file)
            progress(f"[{len(report.made) + len(report.errors)}/{total}] готово: {name}, {best[0]:.1f} нот/с")

    async def main() -> None:
        sem = asyncio.Semaphore(max(1, parallel))
        await asyncio.gather(*(one(i, t, p, sem) for i, (t, p) in enumerate(todo)))

    if todo:
        progress(f"Записываю {total} мелодий ({engine}) по {seconds:.0f} с, дублей на каждую: {takes}, по {parallel} "
                 f"одновременно (примерно {total * takes * (seconds + 8) / max(1, parallel) / 60:.0f} мин)…")
        asyncio.run(main())
        if fatal:
            progress(f"Остановлено: {fatal[0]}")
    return report
