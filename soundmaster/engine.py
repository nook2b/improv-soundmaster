"""Движок воспроизведения: предзагруженные звуки реестра + микшер + оценка времени выхода звука."""

from __future__ import annotations

import bisect
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio.mixer import Mixer, MixerEvent
from .audio.wavio import load_sound
from .clock import Clock
from .journal import Journal
from .registry import Effect, Registry


@dataclass
class LoadReport:
    loaded: list[str]
    missing: list[str]
    errors: list[str]


class Engine:
    def __init__(self, registry: Registry, mixer: Mixer, clock: Clock, journal: Journal):
        self.registry = registry
        self.mixer = mixer
        self.clock = clock
        self.journal = journal
        self.sounds: dict[str, np.ndarray] = {}  # ключ — относительный путь файла
        self._frame_times: deque[tuple[int, float]] = deque(maxlen=4096)
        self._lock = threading.Lock()
        self.voice_info: dict[int, dict] = {}  # voice_id → данные для журнала (решение, источник)

    # ---- загрузка ------------------------------------------------------------------------
    def preload(self) -> LoadReport:
        files = {e.file for e in self.registry.effects.values()}
        files |= {s.file for s in self.registry.music_states.values() if s.file}
        loaded, missing, errors = [], [], []
        for rel in sorted(files):
            path = self.registry.path_of(rel)
            if not path.exists():
                missing.append(rel)
                continue
            try:
                self.sounds[rel] = load_sound(path, self.mixer.sample_rate)
                loaded.append(rel)
            except Exception as e:  # noqa: BLE001 — показать технику любую ошибку файла
                errors.append(f"{rel}: {e}")
        self.journal.write("preload", loaded=len(loaded), missing=missing, errors=errors)
        return LoadReport(loaded, missing, errors)

    def available(self, effect: Effect) -> bool:
        return effect.file in self.sounds

    # ---- воспроизведение -----------------------------------------------------------------
    def play_effect(self, effect: Effect, owner: str, info: dict | None = None) -> int | None:
        data = self.sounds.get(effect.file)
        if data is None:
            return None
        if effect.do_not_restart_if_playing and self.mixer.is_playing(effect.id):
            return None
        vid = self.mixer.play(
            effect.id, data, effect.channel, owner,
            gain_db=effect.gain_db, loop=effect.loop, fade_in=effect.fade_in,
        )
        self.voice_info[vid] = {"effect": effect.id, "owner": owner, "t_command": self.clock.now(), **(info or {})}
        return vid

    def play_buffer(self, effect_id: str, source, bus: str, owner: str, gain_db: float, fade_in: float,
                    info: dict | None = None) -> int:
        vid = self.mixer.play(effect_id, source, bus, owner, gain_db=gain_db, fade_in=fade_in)
        self.voice_info[vid] = {"effect": effect_id, "owner": owner, "t_command": self.clock.now(), **(info or {})}
        return vid

    # ---- время выхода звука --------------------------------------------------------------
    def on_output_block(self, block: np.ndarray, t_out: float) -> None:
        """Вызывается из аудиопотока после render(): запоминаем, когда блок прозвучит."""
        start = self.mixer.frames_rendered - len(block)
        self._frame_times.append((start, t_out))

    def _time_of_frame(self, frame: int) -> float | None:
        times = list(self._frame_times)
        if not times:
            return None
        i = bisect.bisect_right([f for f, _ in times], frame) - 1
        if i < 0:
            return None
        f0, t0 = times[i]
        return t0 + (frame - f0) / self.mixer.sample_rate

    def pump_events(self) -> list[MixerEvent]:
        """Разобрать события микшера в основном потоке и записать их в журнал."""
        out = []
        while self.mixer.events:
            ev = self.mixer.events.popleft()
            info = self.voice_info.get(ev.voice_id, {})
            t_audio = self._time_of_frame(ev.frame)
            if ev.kind == "started":
                latency = None if t_audio is None or "t_command" not in info else round(t_audio - info["t_command"], 4)
                self.journal.write("audio_start", voice_id=ev.voice_id, effect=ev.effect_id, bus=ev.bus,
                                   owner=ev.owner, t_audio_out=None if t_audio is None else round(t_audio, 4),
                                   command_to_audio=latency, request_id=info.get("request_id"))
            else:
                self.journal.write("audio_end", voice_id=ev.voice_id, effect=ev.effect_id, bus=ev.bus, owner=ev.owner)
                self.voice_info.pop(ev.voice_id, None)
            out.append(ev)
        return out
