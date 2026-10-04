"""Распознавание речи (звук фразы → текст).

- `MlxWhisperASR` — основной вариант для Mac M1–M4 (Apple MLX, быстро, без интернета после загрузки).
- `FasterWhisperASR` — Windows/Linux/Intel Mac: на видеокарте NVIDIA (CUDA) или на процессоре.
- `ScriptedASR` — подставной распознаватель для тестов и для прогона с готовой расшифровкой.

Модели скачиваются при первом запуске (нужен интернет один раз).
"""

from __future__ import annotations

import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .vad import Utterance

# Типичные «галлюцинации» Whisper на русском на шуме, смехе и тишине.
HALLUCINATIONS = (
    "продолжение следует",
    "субтитры",
    "редактор субтитров",
    "спасибо за просмотр",
    "подписывайтесь",
    "dimatorzok",
    "корректор",
    "игорь негода",
)


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.lower().replace("ё", "е")).strip()


def clean_text(text: str, prompt: str | None = None) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    low = t.lower()
    if any(h in low for h in HALLUCINATIONS):
        return ""
    if re.fullmatch(r"[\W_]*", t):
        return ""
    if prompt:
        # На шуме и смехе Whisper повторяет подсказку (initial_prompt) вместо речи — это не реплика.
        words = [w[:5] for w in _norm(t).split()]
        pwords = {w[:5] for w in _norm(prompt).split()}
        if words and sum(w in pwords for w in words) / len(words) >= 0.8:
            return ""
    return t


@dataclass
class Transcript:
    text: str
    t_start: float
    t_end: float  # конец фразы во входном звуке
    t_ready: float  # когда текст готов (по часам системы)
    asr_seconds: float
    forced_cut: bool = False


class ASR(ABC):
    name = "base"
    sample_rate = 16000

    @abstractmethod
    def transcribe(self, audio: np.ndarray, t_start: float = 0.0, t_end: float = 0.0) -> str: ...

    def warmup(self) -> None:
        self.transcribe(np.zeros(self.sample_rate // 2, np.float32))


class MlxWhisperASR(ASR):
    name = "mlx"

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo", language: str = "ru",
                 initial_prompt: str | None = None):
        try:
            import mlx_whisper  # type: ignore
        except ImportError as e:
            raise RuntimeError("нет mlx-whisper: pip install -e \".[asr-mac]\" (только Mac с M1–M4)") from e
        self._mlx = mlx_whisper
        self.model = model
        self.language = language
        self.initial_prompt = initial_prompt

    def transcribe(self, audio: np.ndarray, t_start: float = 0.0, t_end: float = 0.0) -> str:
        r = self._mlx.transcribe(audio, path_or_hf_repo=self.model, language=self.language,
                                 initial_prompt=self.initial_prompt, condition_on_previous_text=False,
                                 temperature=0.0)
        return r.get("text", "")


class FasterWhisperASR(ASR):
    name = "faster"

    def __init__(self, model: str = "small", language: str = "ru", initial_prompt: str | None = None,
                 device: str = "auto", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            raise RuntimeError("нет faster-whisper: pip install -e \".[asr-cpu]\"") from e
        # device=auto: видеокарта NVIDIA (CUDA), если есть, иначе процессор.
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        self.initial_prompt = initial_prompt

    def transcribe(self, audio: np.ndarray, t_start: float = 0.0, t_end: float = 0.0) -> str:
        segments, _ = self._model.transcribe(audio, language=self.language, beam_size=1,
                                             initial_prompt=self.initial_prompt,
                                             condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text for s in segments)


class ScriptedASR(ASR):
    """Возвращает заранее известный текст по времени фразы: [(t_start, t_end, text), ...]."""

    name = "scripted"

    def __init__(self, lines: list[tuple[float, float, str]]):
        self.lines = lines

    def transcribe(self, audio: np.ndarray, t_start: float = 0.0, t_end: float = 0.0) -> str:
        parts = [txt for a, b, txt in self.lines if a < t_end and b > t_start]
        return " ".join(parts)

    def warmup(self) -> None:
        pass


def load_transcript_file(path) -> list[tuple[float, float, str]]:
    """Текстовая расшифровка: строки `0:12.5-0:15 текст` или `12.5 15.0 текст`."""

    def sec(v: str) -> float:
        parts = [float(x) for x in v.split(":")]
        out = 0.0
        for p in parts:
            out = out * 60 + p
        return out

    lines = []
    for raw in open(path, encoding="utf-8"):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        m = re.match(r"^([\d:.]+)\s*[-–\s]\s*([\d:.]+)\s+(.*)$", raw)
        if not m:
            raise ValueError(f"не понял строку расшифровки: {raw!r}")
        lines.append((sec(m.group(1)), sec(m.group(2)), m.group(3)))
    return lines


def make_asr(kind: str | None, cfg: dict | None = None) -> ASR | None:
    cfg = cfg or {}
    if not kind or kind == "off":
        return None
    prompt = cfg.get("initial_prompt")
    if kind == "mlx":
        return MlxWhisperASR(cfg.get("mlx_model", "mlx-community/whisper-large-v3-turbo"),
                             cfg.get("language", "ru"), prompt)
    if kind == "faster":
        return FasterWhisperASR(cfg.get("faster_model", "small"), cfg.get("language", "ru"), prompt,
                                cfg.get("faster_device", "auto"), cfg.get("faster_compute", "int8"))
    raise ValueError(f"неизвестный распознаватель: {kind} (mlx | faster | off)")


class ASRWorker:
    """Распознаёт фразы в отдельном потоке, чтобы не задерживать звук и пульт.

    Если распознавание не успевает, старые фразы выбрасываются (устаревший текст бесполезен).
    """

    def __init__(self, asr: ASR, clock, on_transcript: Callable[[Transcript], None], max_queue: int = 4):
        self.asr = asr
        self.clock = clock
        self.on_transcript = on_transcript
        self.q: "queue.Queue[Utterance | None]" = queue.Queue()
        self.max_queue = max_queue
        self.dropped = 0
        self.errors = 0
        self.last_error: str | None = None
        self.state = "idle"  # idle | loading | ready | error
        self._thread = threading.Thread(target=self._run, name="asr", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, u: Utterance) -> None:
        while self.q.qsize() >= self.max_queue:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                break
        self.q.put(u)

    def process(self, u: Utterance) -> Transcript | None:
        t0 = time.monotonic()
        try:
            text = clean_text(self.asr.transcribe(u.audio, u.t_start, u.t_end),
                              getattr(self.asr, "initial_prompt", None))
        except Exception as e:  # noqa: BLE001
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        if not text:
            return None
        return Transcript(text, u.t_start, u.t_end, self.clock.now(), time.monotonic() - t0, u.forced_cut)

    def _run(self) -> None:
        # Загрузка модели в этом же потоке: пульт и звук доступны сразу, распознавание — когда готово.
        self.state = "loading"
        try:
            self.asr.warmup()
            self.state = "ready"
        except Exception as e:  # noqa: BLE001
            self.state = "error"
            self.last_error = f"{type(e).__name__}: {e}"
            return
        while True:
            u = self.q.get()
            if u is None:
                return
            tr = self.process(u)
            if tr:
                self.on_transcript(tr)

    def stop(self) -> None:
        self.q.put(None)
        self._thread.join(timeout=5)
