"""Запись прогона: все каналы входа и опорная дорожка выхода системы, на общих часах.

Аудио-callback'и только кладут блоки в очереди; файлы пишет отдельный поток.
Время первого блока каждой дорожки пишется в журнал, чтобы совместить дорожки и решения.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from ..journal import Journal
from .wavio import WavStreamWriter


class Track:
    def __init__(self, name: str, path: Path, channels: int, sample_rate: int, journal: Journal):
        self.name = name
        self.path = path
        self.channels = channels
        self.sample_rate = sample_rate
        self.journal = journal
        self.q: "queue.Queue[tuple[np.ndarray, float] | None]" = queue.Queue(maxsize=4000)
        self.dropped = 0
        self._writer: WavStreamWriter | None = None
        self._thread = threading.Thread(target=self._run, name=f"rec-{name}", daemon=True)
        self._thread.start()

    def push(self, block: np.ndarray, t: float) -> None:
        try:
            self.q.put_nowait((block.copy(), t))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            block, t = item
            if self._writer is None:
                self._writer = WavStreamWriter(self.path, self.channels, self.sample_rate)
                self.journal.write("track_start", track=self.name, file=self.path.name, t_first_sample=round(t, 4),
                                   channels=self.channels, sample_rate=self.sample_rate)
            self._writer.write(block)
        if self._writer:
            self._writer.close()
            self.journal.write("track_end", track=self.name, frames=self._writer.frames, dropped=self.dropped)

    def close(self) -> None:
        self.q.put(None)
        self._thread.join(timeout=5)


class Recorder:
    def __init__(self, directory: str | Path, journal: Journal):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.journal = journal
        self.tracks: dict[str, Track] = {}

    def track(self, name: str, channels: int, sample_rate: int) -> Track:
        if name not in self.tracks:
            self.tracks[name] = Track(name, self.dir / f"{name}.wav", channels, sample_rate, self.journal)
        return self.tracks[name]

    def close(self) -> None:
        for t in self.tracks.values():
            t.close()
