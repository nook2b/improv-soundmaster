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
        self.error: str | None = None  # ошибка записи (поток жив и разбирает очередь)
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
            if self.error is not None:
                continue  # запись уже сломалась: очередь разбираем, чтобы push и close не вставали
            block, t = item
            try:
                if self._writer is None:
                    self._writer = WavStreamWriter(self.path, self.channels, self.sample_rate)
                    self.journal.write("track_start", track=self.name, file=self.path.name,
                                       t_first_sample=round(t, 4), channels=self.channels,
                                       sample_rate=self.sample_rate)
                self._writer.write(block)
            except Exception as e:  # noqa: BLE001 — например, кончился диск: шоу идёт дальше без записи
                self.error = f"{type(e).__name__}: {e}"
                self.journal.write("track_error", track=self.name, error=self.error)
        if self._writer:
            try:
                self._writer.close()
            except Exception as e:  # noqa: BLE001
                self.error = self.error or f"{type(e).__name__}: {e}"
            self.journal.write("track_end", track=self.name, frames=self._writer.frames, dropped=self.dropped,
                               error=self.error)

    def close(self, timeout: float = 5.0) -> None:
        """Не блокируется навсегда: если поток записи завис, очередь освобождается от блоков."""
        try:
            self.q.put(None, timeout=timeout)
        except queue.Full:
            while True:  # поток не разбирает очередь — выбрасываем блоки, чтобы поместился сигнал конца
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    break
            try:
                self.q.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=timeout)


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
