"""Журнал решений и событий (JSONL, одна запись на строку)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .clock import Clock


class Journal:
    def __init__(self, path: str | Path | None, clock: Clock, base: dict[str, Any] | None = None):
        self.clock = clock
        self.base = dict(base or {})
        self.records: list[dict[str, Any]] = []  # последние записи для пульта и тестов
        self._lock = threading.Lock()
        self._fh = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        rec = {"t": round(self.clock.now(), 4), "kind": kind, **self.base, **fields}
        with self._lock:
            self.records.append(rec)
            if len(self.records) > 1000:
                del self.records[:500]
            if self._fh:
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._fh.flush()
        return rec

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
