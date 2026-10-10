"""Пул записанных мелодий: по нескольку на жанр, каждая — один инструмент.

Зачем: ручные кнопки жанров на пульте (повторное нажатие — следующая мелодия) и резерв, когда
генератор недоступен (нет интернета). Записывает их Lyria RealTime по тем же пресетам словаря, что
и живая музыка (`python -m soundmaster make-music-pool`), — звучат так же, только без сети.

Каждая мелодия — бесшовная петля: хвост записи плавно наложен на начало, громкость выровнена.
"""

from __future__ import annotations

import asyncio
import os
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


def write_flac(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.flac")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                    "-c:a", "flac", "-sample_fmt", "s16", str(tmp)],
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


@dataclass
class PoolReport:
    made: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def select(registry: Registry, only: str | None = None) -> list[PoolTrack]:
    """Все мелодии пула или только жанр (`noir`) / одна мелодия (`noir/2`)."""
    tracks = [t for ts in registry.pool.values() for t in ts]
    if not only:
        return tracks
    genre, _, n = only.partition("/")
    if genre not in registry.pool:
        raise KeyError(f"в пуле нет жанра {genre!r}; есть: {', '.join(registry.pool)}")
    return [t for t in registry.pool[genre] if not n or str(t.n) == n]


def make_pool(registry: Registry, seconds: float = 60.0, parallel: int = 4, only: str | None = None,
              force: bool = False, progress: Callable[[str], None] = print,
              recorder: Callable | None = None) -> PoolReport:
    """Записать недостающие мелодии пула (force — перезаписать). recorder — для тестов."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("нужен ffmpeg (Mac: brew install ffmpeg)")
    key = os.environ.get("GEMINI_API_KEY", "")
    if recorder is None and not key:
        raise RuntimeError("нужен GEMINI_API_KEY (Lyria)")
    rec = recorder or record_lyria
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
    total = len(todo)

    async def one(i: int, t: PoolTrack, path: Path, sem: asyncio.Semaphore) -> None:
        async with sem:
            name = f"{t.file} ({vocab.label(t.style)})"
            for attempt in (1, 2, 3):
                try:
                    raw = await rec(vocab.preset(t.style), SKIP_START + seconds + LOOP_XFADE, key)
                    x = make_loop(raw[int(SKIP_START * SR):], seconds)
                    write_flac(path, normalize(x))
                    report.made.append(t.file)
                    progress(f"[{len(report.made) + len(report.errors)}/{total}] готово: {name}")
                    return
                except Exception as e:  # noqa: BLE001 — сеть/Lyria: пробуем ещё, потом пропускаем
                    err = f"{type(e).__name__}: {e}"
                    if attempt < 3:
                        progress(f"повтор {attempt}: {name} — {err}")
                        await asyncio.sleep(2 * attempt)
            report.errors.append((t.file, err))
            progress(f"[{len(report.made) + len(report.errors)}/{total}] ОШИБКА: {name} — {err}")

    async def main() -> None:
        sem = asyncio.Semaphore(max(1, parallel))
        await asyncio.gather(*(one(i, t, p, sem) for i, (t, p) in enumerate(todo)))

    if todo:
        progress(f"Записываю {total} мелодий по {seconds:.0f} с, по {parallel} одновременно "
                 f"(примерно {total * (seconds + 8) / max(1, parallel) / 60:.0f} мин)…")
        asyncio.run(main())
    return report
