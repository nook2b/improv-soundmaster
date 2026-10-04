"""Единые монотонные часы системы (секунды от старта)."""

from __future__ import annotations

import time


class Clock:
    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self.wall_start = time.time()

    def now(self) -> float:
        return time.monotonic() - self._t0


class FakeClock(Clock):
    """Ручные часы для тестов и стенда повторов."""

    def __init__(self, t: float = 0.0) -> None:
        super().__init__()
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt
