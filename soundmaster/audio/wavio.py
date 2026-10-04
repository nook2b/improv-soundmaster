"""Чтение и запись WAV без тяжёлых зависимостей.

Чтение: если установлен `soundfile`, используется он (любой WAV/AIFF/FLAC); иначе stdlib `wave`
(PCM 8/16/24/32 бит). Все звуки приводятся к float32 стерео нужной частоты.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def _read_stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        i = np.where(i & 0x800000, i - 0x1000000, i)
        data = i.astype(np.float32) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: неподдерживаемая разрядность {width * 8} бит")
    return data.reshape(-1, ch), sr


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    try:
        import soundfile  # type: ignore

        data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        return data, sr
    except ImportError:
        pass
    try:
        return _read_stdlib(path)
    except wave.Error as e:
        raise ValueError(
            f"{path}: {e}. Для float-WAV, AIFF и FLAC установите пакет soundfile"
        ) from e


def to_stereo(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def resample_linear(data: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or len(data) == 0:
        return data
    n_out = int(round(len(data) * sr_to / sr_from))
    x_old = np.arange(len(data), dtype=np.float64)
    x_new = np.linspace(0, len(data) - 1, n_out)
    return np.stack([np.interp(x_new, x_old, data[:, c]) for c in range(data.shape[1])], axis=1).astype(
        np.float32
    )


def load_sound(path: str | Path, sample_rate: int) -> np.ndarray:
    data, sr = read_audio(path)
    data = to_stereo(data.astype(np.float32, copy=False))
    return np.ascontiguousarray(resample_linear(data, sr, sample_rate), dtype=np.float32)


def write_wav(path: str | Path, data: np.ndarray, sample_rate: int) -> None:
    """Запись float32 [-1, 1] в 16-битный PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 1:
        data = data[:, None]
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


class WavStreamWriter:
    """Потоковая запись 16-битного WAV (для записи входа и опорной дорожки выхода)."""

    def __init__(self, path: str | Path, channels: int, sample_rate: int):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._w = wave.open(str(path), "wb")
        self._w.setnchannels(channels)
        self._w.setsampwidth(2)
        self._w.setframerate(sample_rate)
        self.frames = 0

    def write(self, block: np.ndarray) -> None:
        pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2")
        self._w.writeframes(pcm.tobytes())
        self.frames += len(block)

    def close(self) -> None:
        self._w.close()
