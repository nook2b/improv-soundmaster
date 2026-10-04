"""Импорт звуков из папки: любые форматы (mp3, mp4, m4a, wav …) → WAV по путям из реестра.

Файл сопоставляется по имени (без расширения) с id эффекта или музыкального состояния:
`doorbell.mp3` → эффект doorbell → sounds/sfx/doorbell.wav. Через ffmpeg:
- 48 кГц, стерео, 16 бит;
- у коротких эффектов обрезается тишина в начале (иначе звук «опаздывает» на эту тишину);
- громкость выравнивается (EBU R128): эффекты и отбивки громче, атмосфера и музыка — тише,
  чтобы файлы из разных источников звучали ровно; дальше — gain_db из реестра.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .registry import Registry

AUDIO_EXT = {".wav", ".mp3", ".mp4", ".m4a", ".aac", ".flac", ".ogg", ".aif", ".aiff", ".wma", ".opus"}
LOUDNESS = {"sfx": -18.0, "sting": -16.0, "ambience": -28.0, "music": -24.0}  # LUFS


@dataclass
class ImportReport:
    imported: list[tuple[str, Path]] = field(default_factory=list)  # (id, куда записан)
    unknown: list[Path] = field(default_factory=list)  # файл, которому нет id в реестре
    skipped: list[tuple[str, Path]] = field(default_factory=list)  # уже есть (без --force)
    errors: list[tuple[Path, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # id реестра, для которых файла так и нет


def _targets(registry: Registry) -> dict[str, tuple[str, str]]:
    """id → (путь в реестре, канал)."""
    out = {e.id: (e.file, e.channel) for e in registry.effects.values()}
    out.update({m.id: (m.file, m.channel) for m in registry.music_states.values() if m.file})
    return out


def convert(src: Path, dst: Path, channel: str) -> None:
    filters = []
    if channel in ("sfx", "sting"):
        # Тишина в начале (тише −50 дБ) срезается; 5 мс запаса, чтобы не съесть атаку.
        filters.append("silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.005")
    filters.append(f"loudnorm=I={LOUDNESS.get(channel, -20.0)}:TP=-1.5:LRA=20")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-af", ",".join(filters),
           "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(r.stderr.strip()[-300:] or "ffmpeg: ошибка")
    tmp.replace(dst)


def import_sounds(registry: Registry, folder: Path, force: bool = False) -> ImportReport:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("нужен ffmpeg (Windows: winget install ffmpeg; Mac: brew install ffmpeg)")
    rep = ImportReport()
    targets = _targets(registry)
    for src in sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT):
        sid = src.stem.strip().lower().replace(" ", "_").replace("-", "_")
        if sid not in targets:
            rep.unknown.append(src)
            continue
        rel, channel = targets[sid]
        dst = registry.path_of(rel)
        if dst.exists() and not force:
            rep.skipped.append((sid, dst))
            continue
        try:
            convert(src, dst, channel)
            rep.imported.append((sid, dst))
        except Exception as e:  # noqa: BLE001 — один плохой файл не мешает остальным
            rep.errors.append((src, str(e)))
    rep.missing = [sid for sid, (rel, _) in targets.items() if not registry.path_of(rel).exists()]
    return rep
