"""Нарезка звука на фразы по энергии (простой VAD с адаптивным уровнем шума).

На сцене речь почти непрерывная, поэтому фраза принудительно режется по max_utterance:
иначе распознавание и решения ждали бы паузы слишком долго.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Utterance:
    audio: np.ndarray  # float32 моно, sample_rate VAD
    t_start: float  # время первого отсчёта (по часам системы или записи)
    t_end: float
    forced_cut: bool = False  # разрезана по длине, а не по паузе


class EnergyVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        threshold_db: float = 10.0,  # над уровнем шума
        min_speech_ms: int = 150,
        hangover_ms: int = 450,  # пауза, после которой фраза закончена
        preroll_ms: int = 200,
        max_utterance_s: float = 6.0,
        min_utterance_s: float = 0.25,
        floor_db: float = -60.0,
    ):
        self.sr = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.threshold_db = threshold_db
        self.min_speech = max(1, min_speech_ms // frame_ms)
        self.hangover = max(1, hangover_ms // frame_ms)
        self.preroll = deque(maxlen=max(1, preroll_ms // frame_ms))
        self.max_frames = int(max_utterance_s * 1000 / frame_ms)
        self.min_frames = int(min_utterance_s * 1000 / frame_ms)
        self.noise_db = floor_db + 20.0
        self.floor_db = floor_db
        self._buf = np.zeros(0, np.float32)
        self._t_buf = 0.0  # время первого отсчёта в _buf
        self._speech: list[np.ndarray] = []
        self._t_speech = 0.0
        self._voiced_run = 0
        self._silent_run = 0
        self.in_speech = False
        self.level_db = floor_db

    def _emit(self, t_end: float, forced: bool) -> Utterance | None:
        frames = self._speech
        self._speech = []
        self.in_speech = False
        self._silent_run = 0
        self._voiced_run = 0
        if len(frames) < self.min_frames:
            return None
        return Utterance(np.concatenate(frames), self._t_speech, t_end, forced)

    def push(self, block: np.ndarray, t_block: float) -> list[Utterance]:
        """Добавить блок (моно float32) со временем его первого отсчёта; вернуть законченные фразы."""
        if len(self._buf) == 0:
            self._t_buf = t_block
        self._buf = np.concatenate([self._buf, block.astype(np.float32, copy=False)])
        out: list[Utterance] = []
        while len(self._buf) >= self.frame:
            fr = self._buf[: self.frame]
            self._buf = self._buf[self.frame:]
            t_fr = self._t_buf
            self._t_buf += self.frame / self.sr
            db = 20 * np.log10(float(np.sqrt(np.mean(fr * fr))) + 1e-9)
            self.level_db = db
            voiced = db > max(self.noise_db + self.threshold_db, self.floor_db)
            if not self.in_speech:
                # Уровень шума следит за тишиной: быстро вниз, медленно вверх.
                rate = 0.2 if db < self.noise_db else 0.01
                self.noise_db += (db - self.noise_db) * rate
                self.preroll.append((t_fr, fr))
                self._voiced_run = self._voiced_run + 1 if voiced else 0
                if self._voiced_run >= self.min_speech:
                    self.in_speech = True
                    self._speech = [f for _, f in self.preroll]
                    self._t_speech = self.preroll[0][0]
                    self.preroll.clear()
                    self._silent_run = 0
                continue
            self._speech.append(fr)
            self._silent_run = 0 if voiced else self._silent_run + 1
            t_end = t_fr + self.frame / self.sr
            if self._silent_run >= self.hangover:
                u = self._emit(t_end, False)
                if u:
                    out.append(u)
            elif len(self._speech) >= self.max_frames:
                u = self._emit(t_end, True)
                if u:
                    out.append(u)
                # Речь продолжается: следующая фраза начинается сразу.
                self.in_speech = True
                self._t_speech = t_end
        return out

    def flush(self, t_end: float) -> list[Utterance]:
        if self.in_speech:
            u = self._emit(t_end, False)
            return [u] if u else []
        return []
