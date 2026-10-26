"""Автоотправка разметки на GitHub: Ивану не нужно набирать git add / commit / pull / push.

Отправляется только папка data/scenecut (рамки людей и отметки — числа, без видео). Перед отправкой
забираются чужие изменения (git pull — Claude правит код в той же ветке). Если что-то пошло не так
(нет сети, конфликт), ничего не теряется: изменения остаются в папке и уйдут при следующей попытке.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyncResult:
    ok: bool
    message: str
    at: float


class GitSync:
    def __init__(self, data_dir: Path, enabled: bool = True):
        self.data_dir = data_dir.resolve()
        self.enabled = enabled and shutil.which("git") is not None
        self.root: Path | None = None
        if self.enabled:
            out = self._git("rev-parse", "--show-toplevel", cwd=self.data_dir if self.data_dir.exists() else Path.cwd())
            self.enabled = out.returncode == 0
            self.root = Path(out.stdout.strip()) if self.enabled else None
        self.last: SyncResult | None = None
        self._lock = threading.Lock()

    def _git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=str(cwd or self.root), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120)

    def changed(self) -> bool:
        if not self.enabled:
            return False
        out = self._git("status", "--porcelain", "--", str(self.data_dir))
        return bool(out.stdout.strip())

    def push(self, message: str = "Разметка монтажей (автоматически)") -> SyncResult:
        """Закоммитить data/scenecut (если есть изменения), забрать ветку и отправить. Не бросает исключений."""
        if not self.enabled:
            return SyncResult(False, "git не найден или папка не в репозитории — отправьте вручную", time.time())
        with self._lock:
            try:
                res = self._push(message)
            except (OSError, subprocess.SubprocessError) as e:
                res = SyncResult(False, f"{type(e).__name__}: {e}", time.time())
            self.last = res
            return res

    def _push(self, message: str) -> SyncResult:
        now = time.time()
        if self.changed():
            self._git("add", "--", str(self.data_dir))
            c = self._git("commit", "-m", message, "--", str(self.data_dir))
            if c.returncode != 0 and "nothing to commit" not in (c.stdout + c.stderr):
                return SyncResult(False, "не удалось сохранить: " + _tail(c), now)
        pull = self._git("pull", "--no-rebase", "--no-edit")
        if pull.returncode != 0:
            self._git("merge", "--abort")  # конфликт — не оставляем репозиторий в полуслиянии
            return SyncResult(False, "не удалось забрать изменения с GitHub: " + _tail(pull), now)
        ahead = self._git("rev-list", "--count", "@{u}..HEAD")
        if ahead.returncode == 0 and ahead.stdout.strip() == "0":
            return SyncResult(True, "всё отправлено", now)
        p = self._git("push")
        if p.returncode != 0:
            return SyncResult(False, "не удалось отправить: " + _tail(p), now)
        return SyncResult(True, "отправлено", now)


def _tail(p: subprocess.CompletedProcess) -> str:
    text = (p.stderr or p.stdout or "").strip().splitlines()
    return text[-1][:200] if text else f"код {p.returncode}"
