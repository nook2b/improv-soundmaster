"""Скачивание записи по ссылке (YouTube, Google Drive и др.) через yt-dlp для прогона.

На Mac: brew install yt-dlp. Файлы кэшируются в recordings/downloads и повторно не скачиваются.
Записи с актёрами и зрителями — только с согласия команды; в git они не попадают.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip()))


def download(url: str, directory: Path, max_height: int = 720) -> Path:
    exe = shutil.which("yt-dlp")
    if not exe:
        raise RuntimeError("для ссылок нужен yt-dlp. На Mac: brew install yt-dlp")
    directory.mkdir(parents=True, exist_ok=True)
    fmt = f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"
    template = str(directory / "%(id)s.%(ext)s")
    out = subprocess.run([exe, "-f", fmt, "--merge-output-format", "mp4", "-o", template,
                          "--print", "after_move:filepath", "--no-playlist", url],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"не удалось скачать {url}: {out.stderr.strip()[-400:]}")
    path = Path(out.stdout.strip().splitlines()[-1])
    if not path.exists():
        raise RuntimeError(f"yt-dlp не сообщил путь к файлу: {out.stdout[-300:]}")
    return path
