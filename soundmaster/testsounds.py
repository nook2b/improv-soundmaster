"""Синтетические тестовые звуки для всех файлов реестра — чтобы пульт работал до появления настоящих.

Существующие файлы не перезаписываются (кроме --force).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio.wavio import write_wav
from .registry import Registry

SR = 48000
rng = np.random.default_rng(7)


def _t(sec: float) -> np.ndarray:
    return np.arange(int(sec * SR)) / SR


def _env(n: int, attack: float = 0.005, decay: float = 0.2) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / max(decay, 1e-4))


def _noise(sec: float) -> np.ndarray:
    return rng.standard_normal(int(sec * SR))


def _lowpass(x: np.ndarray, a: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):  # простой однополюсный фильтр; звуки короткие
        acc += a * (v - acc)
        y[i] = acc
    return y


def shutter() -> np.ndarray:
    a = _noise(0.03) * _env(int(0.03 * SR), 0.001, 0.008)
    gap = np.zeros(int(0.06 * SR))
    b = _noise(0.04) * _env(int(0.04 * SR), 0.001, 0.012)
    return 0.6 * np.concatenate([a, gap, b])


def doorbell() -> np.ndarray:
    def tone(f, sec):
        t = _t(sec)
        return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.01 * f * t)) * _env(len(t), 0.005, 0.5)
    return 0.4 * np.concatenate([tone(659, 0.6), tone(523, 0.9)])


def phone_ring() -> np.ndarray:
    t = _t(1.2)
    ring = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)
    gate = (np.sin(2 * np.pi * 20 * t) > 0).astype(float)
    one = 0.2 * ring * gate
    return np.concatenate([one, np.zeros(int(0.4 * SR)), one])


def knock() -> np.ndarray:
    def k():
        t = _t(0.12)
        return (np.sin(2 * np.pi * 180 * t) + 0.5 * _noise(0.12)) * _env(len(t), 0.001, 0.025)
    gap = np.zeros(int(0.15 * SR))
    return 0.6 * np.concatenate([k(), gap, k(), gap, k()])


def hit() -> np.ndarray:
    t = _t(0.35)
    body = np.sin(2 * np.pi * (90 * np.exp(-t * 8) + 50) * t)
    return 0.8 * (body + 0.4 * _noise(0.35) * np.exp(-t * 40)) * _env(len(t), 0.001, 0.12)


def fall() -> np.ndarray:
    t = _t(0.8)
    thud = np.sin(2 * np.pi * (60 * np.exp(-t * 3) + 30) * t) * _env(len(t), 0.002, 0.25)
    rattle = _lowpass(_noise(0.8), 0.2) * np.exp(-t * 6)
    return 0.8 * (thud + 0.5 * rattle)


def magic() -> np.ndarray:
    t = _t(1.5)
    f = 600 + 900 * t
    return 0.25 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(2 * np.pi * 12 * t) ** 2 * _env(len(t), 0.05, 0.8)


def gunshot() -> np.ndarray:
    t = _t(0.9)
    return 0.9 * _noise(0.9) * _env(len(t), 0.0005, 0.09)


def sting() -> np.ndarray:
    t = _t(1.6)
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0, 523.3))
    return 0.2 * chord * _env(len(t), 0.01, 0.6)


def music_loop(root: float, chord: list[int], bpm: float, sec: float = 16.0) -> np.ndarray:
    t = _t(sec)
    sig = sum(np.sin(2 * np.pi * root * 2 ** (s / 12) * t) for s in chord) / len(chord)
    pulse = 0.6 + 0.4 * np.cos(np.pi * bpm / 60 * t) ** 2
    fade = np.minimum(1, np.minimum(t, sec - t) / 0.05)  # без щелчка на шве петли
    return 0.25 * sig * pulse * fade


def ambience(kind: str, sec: float = 12.0) -> np.ndarray:
    n = _noise(sec)
    if kind == "rain":
        x = _lowpass(n, 0.5) * 0.15
    else:  # гул зала
        x = _lowpass(n, 0.05) * 0.5
    t = _t(sec)
    return x * np.minimum(1, np.minimum(t, sec - t) / 0.05)


SYNTHS = {
    "sfx/camera_shutter.wav": shutter,
    "sfx/doorbell.wav": doorbell,
    "sfx/phone_ring.wav": phone_ring,
    "sfx/knock.wav": knock,
    "sfx/hit.wav": hit,
    "sfx/fall.wav": fall,
    "sfx/magic.wav": magic,
    "sfx/gunshot.wav": gunshot,
    "sting/scene_end.wav": sting,
    "music/romantic.wav": lambda: music_loop(220, [0, 4, 7, 11], 60),
    "music/tension.wav": lambda: music_loop(110, [0, 1, 6], 90),
    "ambience/restaurant.wav": lambda: ambience("room"),
    "ambience/rain.wav": lambda: ambience("rain"),
}


def make_test_sounds(registry: Registry, force: bool = False) -> list[Path]:
    wanted = {e.file for e in registry.effects.values()} | {
        s.file for s in registry.music_states.values() if s.file
    }
    written = []
    for rel in sorted(wanted):
        path = registry.path_of(rel)
        if path.exists() and not force:
            continue
        synth = SYNTHS.get(rel)
        if synth is None:  # неизвестный файл — короткий сигнал-заглушка
            t = _t(0.3)
            data = 0.3 * np.sin(2 * np.pi * 880 * t) * _env(len(t), 0.005, 0.1)
        else:
            data = synth()
        write_wav(path, np.asarray(data, np.float32), SR)
        written.append(path)
    return written
