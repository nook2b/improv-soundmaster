"""Профили входа звука (config/inputs/*.yaml).

Профиль описывает, какое устройство слушаем и что физически приходит на каждый его канал.
Сейчас — рекордер у сцены (всё вместе, включая колонки). Потом — петлички через микшер или
многоканальный аудиоинтерфейс. Остальная программа смотрит только на роли каналов, поэтому
смена оборудования — это новый YAML-файл, а не изменение кода.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROLES = {
    "stage_mix": "общий звук сцены (рекордер/микрофон у сцены) — слышит и колонки",
    "actor_mic": "петличка одного актёра",
    "actors_aux": "отдельный выход микшера только с микрофонами актёров",
    "room": "микрофон зала (реакция зрителей)",
    "reference": "петля собственного выхода системы (опорная дорожка)",
    "unused": "канал не используется",
}

CLEAN_ROLES = ("actors_aux", "actor_mic")


class InputProfileError(ValueError):
    pass


@dataclass
class InputChannel:
    index: int  # номер канала устройства, с 0
    role: str
    name: str = ""


@dataclass
class InputProfile:
    name: str
    device: str | None
    sample_rate: int
    channels: list[InputChannel] = field(default_factory=list)
    notes: str = ""

    @property
    def device_channels(self) -> int:
        return max(c.index for c in self.channels) + 1

    def indices(self, *roles: str) -> list[int]:
        return [c.index for c in self.channels if c.role in roles]

    @property
    def clean_actor_feed(self) -> bool:
        """Есть ли звук актёров без звуков системы из колонок (электрически)."""
        return bool(self.indices(*CLEAN_ROLES))

    @property
    def ai_feed_roles(self) -> tuple[str, ...]:
        """Что отдавать AI и распознаванию речи: самый чистый доступный источник."""
        if self.indices("actors_aux"):
            return ("actors_aux",)
        if self.indices("actor_mic"):
            return ("actor_mic",)
        return ("stage_mix",)

    def ai_feed(self, block: np.ndarray) -> np.ndarray:
        """Моно-сигнал для AI/ASR из блока устройства (frames × device_channels)."""
        idx = self.indices(*self.ai_feed_roles)
        if not idx:
            return np.zeros(len(block), np.float32)
        return block[:, idx].mean(axis=1).astype(np.float32)

    def room_feed(self, block: np.ndarray) -> np.ndarray | None:
        idx = self.indices("room")
        return block[:, idx].mean(axis=1).astype(np.float32) if idx else None

    def describe(self) -> str:
        lines = [f"Профиль входа: {self.name} (устройство: {self.device or 'по умолчанию'}, {self.sample_rate} Гц)"]
        for c in self.channels:
            lines.append(f"  канал {c.index + 1}: {c.role} — {c.name or ROLES[c.role]}")
        lines.append(f"  чистый звук актёров: {'да' if self.clean_actor_feed else 'нет — общий микрофон слышит и колонки'}")
        lines.append(f"  вход для AI: {', '.join(self.ai_feed_roles)}")
        return "\n".join(lines)


def load_input_profile(path: str | Path) -> InputProfile:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    channels = []
    for raw in data.get("channels") or []:
        role = raw.get("role")
        if role not in ROLES:
            raise InputProfileError(f"{path.name}: роль {role!r} не из {list(ROLES)}")
        channels.append(InputChannel(index=int(raw["channel"]) - 1, role=role, name=raw.get("name", "")))
    if not channels:
        raise InputProfileError(f"{path.name}: нет каналов")
    idx = [c.index for c in channels]
    if len(set(idx)) != len(idx) or min(idx) < 0:
        raise InputProfileError(f"{path.name}: номера каналов повторяются или меньше 1")
    return InputProfile(
        name=data.get("name", path.stem),
        device=data.get("device"),
        sample_rate=int(data.get("sample_rate", 48000)),
        channels=channels,
        notes=data.get("notes", ""),
    )
