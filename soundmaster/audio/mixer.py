"""Программный микшер: голоса на шинах sfx / music / ambience / sting.

`Mixer.render()` вызывается из аудио-callback'а и не должен блокироваться: команды из других
потоков приходят через `deque` (атомарные append/popleft), файлы не читаются, журнал не пишется.
События (старт/конец голоса) складываются в очередь, которую разбирает основной поток.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

BUSES = ("sfx", "music", "ambience", "sting")


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


class BufferSource:
    """Предзагруженный звук в памяти."""

    def __init__(self, data: np.ndarray, loop: bool = False):
        self.data = data
        self.loop = loop
        self.pos = 0

    def read(self, n: int) -> tuple[np.ndarray, bool]:
        out = np.zeros((n, 2), np.float32)
        filled = 0
        total = len(self.data)
        if total == 0:
            return out, True
        while filled < n:
            take = min(n - filled, total - self.pos)
            out[filled : filled + take] = self.data[self.pos : self.pos + take]
            filled += take
            self.pos += take
            if self.pos >= total:
                if not self.loop:
                    return out, True
                self.pos = 0
        return out, False


class StreamBuffer:
    """Кольцевой буфер для потоковой музыки (генератор пишет, микшер читает)."""

    def __init__(self, max_seconds: float, sample_rate: int):
        self.max_frames = int(max_seconds * sample_rate)
        self._chunks: deque[np.ndarray] = deque()
        self._frames = 0
        self._lock = threading.Lock()
        self.underruns = 0
        self.closed = False

    @property
    def frames(self) -> int:
        return self._frames

    def push(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(chunk.astype(np.float32, copy=False))
            self._frames += len(chunk)
            while self._frames > self.max_frames and self._chunks:
                old = self._chunks.popleft()
                self._frames -= len(old)

    def read(self, n: int) -> tuple[np.ndarray, bool]:
        out = np.zeros((n, 2), np.float32)
        filled = 0
        # Короткая блокировка: только перекладывание ссылок/срезов.
        with self._lock:
            while filled < n and self._chunks:
                c = self._chunks[0]
                take = min(n - filled, len(c))
                out[filled : filled + take] = c[:take]
                filled += take
                if take == len(c):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = c[take:]
                self._frames -= take
        if filled < n and not self.closed:
            self.underruns += 1
        return out, self.closed and self._frames == 0


class Voice:
    def __init__(
        self,
        voice_id: int,
        effect_id: str,
        bus: str,
        owner: str,
        source,
        gain: float,
        fade_in_samples: int,
    ):
        self.id = voice_id
        self.effect_id = effect_id
        self.bus = bus
        self.owner = owner  # "manual" | "ai"
        self.source = source
        self.gain = gain
        self.env = 0.0 if fade_in_samples > 0 else 1.0
        self.env_target = 1.0
        self.env_step = 1.0 / fade_in_samples if fade_in_samples > 0 else 0.0
        self.stopping = False
        self.done = False
        self.started = False

    def fade_to(self, target: float, samples: int) -> None:
        self.env_target = target
        self.env_step = abs(target - self.env) / max(samples, 1)
        if target <= 0.0:
            self.stopping = True
            if samples <= 0:
                self.env = 0.0

    def render(self, n: int) -> np.ndarray:
        block, exhausted = self.source.read(n)
        if self.env != self.env_target and self.env_step > 0:
            sign = 1.0 if self.env_target > self.env else -1.0
            ramp = self.env + sign * self.env_step * np.arange(1, n + 1, dtype=np.float32)
            ramp = np.minimum(ramp, self.env_target) if sign > 0 else np.maximum(ramp, self.env_target)
            self.env = float(ramp[-1])
            block *= ramp[:, None]
        elif self.env != 1.0:
            block *= self.env
        if self.gain != 1.0:
            block *= self.gain
        if exhausted or (self.stopping and self.env <= 0.0):
            self.done = True
        return block


@dataclass
class BusConfig:
    gain_db: float = 0.0
    max_simultaneous: int = 0  # 0 — без ограничения
    duck_under: tuple[str, ...] = ()
    duck_db: float = 0.0


DEFAULT_BUSES = {
    "sfx": BusConfig(max_simultaneous=2),
    "music": BusConfig(gain_db=-3.0, max_simultaneous=1, duck_under=("sfx", "sting"), duck_db=-8.0),
    "ambience": BusConfig(gain_db=-6.0, max_simultaneous=1, duck_under=("sfx", "sting"), duck_db=-6.0),
    "sting": BusConfig(max_simultaneous=1),
}


@dataclass
class MixerEvent:
    kind: str  # "started" | "ended"
    voice_id: int
    effect_id: str
    bus: str
    owner: str
    frame: int  # номер первого кадра блока, в котором произошло событие
    offset: int = 0


class Mixer:
    def __init__(
        self,
        sample_rate: int = 48000,
        buses: dict[str, BusConfig] | None = None,
        ceiling_db: float = -1.0,
        replace_fade: float = 0.03,
        duck_time: float = 0.15,
    ):
        self.sample_rate = sample_rate
        self.buses = dict(buses or DEFAULT_BUSES)
        self.bus_gain = {b: db_to_gain(c.gain_db) for b, c in self.buses.items()}
        self.duck_env = {b: 1.0 for b in self.buses}
        self.master_gain = 1.0
        self.ceiling = db_to_gain(ceiling_db)
        self.replace_fade = int(replace_fade * sample_rate)
        self.duck_time = duck_time
        self.voices: list[Voice] = []
        self.frames_rendered = 0
        self.events: deque[MixerEvent] = deque()
        self.playing: frozenset[tuple[int, str, str, str]] = frozenset()  # снимок для других потоков
        self.limited_blocks = 0
        self._cmds: deque[Callable[[], None]] = deque()
        self._ids = itertools.count(1)

    # ---- API для других потоков -------------------------------------------------------
    def play(
        self,
        effect_id: str,
        data_or_source,
        bus: str,
        owner: str,
        gain_db: float = 0.0,
        loop: bool = False,
        fade_in: float = 0.0,
    ) -> int:
        if bus not in self.buses:
            raise ValueError(f"неизвестная шина {bus}")
        vid = next(self._ids)
        source = (
            data_or_source
            if hasattr(data_or_source, "read")
            else BufferSource(data_or_source, loop=loop)
        )
        voice = Voice(vid, effect_id, bus, owner, source, db_to_gain(gain_db), int(fade_in * self.sample_rate))
        self._cmds.append(lambda: self._start(voice))
        return vid

    def stop(self, voice_id: int | None = None, *, effect_id: str | None = None, bus: str | None = None,
             owner: str | None = None, fade: float = 0.03) -> None:
        samples = int(fade * self.sample_rate)

        def cmd() -> None:
            for v in self.voices:
                if voice_id is not None and v.id != voice_id:
                    continue
                if effect_id is not None and v.effect_id != effect_id:
                    continue
                if bus is not None and v.bus != bus:
                    continue
                if owner is not None and v.owner != owner:
                    continue
                v.fade_to(0.0, samples)

        self._cmds.append(cmd)

    def fade(self, voice_id: int, gain: float, seconds: float) -> None:
        """Плавно увести громкость голоса к gain (0..1), не останавливая его."""
        samples = int(seconds * self.sample_rate)

        def cmd() -> None:
            for v in self.voices:
                if v.id == voice_id and not v.stopping:
                    v.fade_to(max(gain, 1e-4), samples)

        self._cmds.append(cmd)

    def stop_all(self, fade: float = 0.03) -> None:
        self.stop(fade=fade)

    def set_bus_gain_db(self, bus: str, db: float) -> None:
        g = db_to_gain(db)
        self._cmds.append(lambda: self.bus_gain.__setitem__(bus, g))

    def set_master_gain_db(self, db: float) -> None:
        g = db_to_gain(db)
        self._cmds.append(lambda: setattr(self, "master_gain", g))

    def is_playing(self, effect_id: str) -> bool:
        return any(e == effect_id for _, e, _, _ in self.playing)

    # ---- аудиопоток -------------------------------------------------------------------
    def _start(self, voice: Voice) -> None:
        limit = self.buses[voice.bus].max_simultaneous
        if limit:
            active = [v for v in self.voices if v.bus == voice.bus and not v.stopping]
            for old in active[: max(0, len(active) - limit + 1)]:
                old.fade_to(0.0, self.replace_fade)
        self.voices.append(voice)

    def render(self, n: int) -> np.ndarray:
        while self._cmds:
            self._cmds.popleft()()

        out = np.zeros((n, 2), np.float32)
        bus_out = {b: None for b in self.buses}
        for v in self.voices:
            block = v.render(n)
            if not v.started:
                v.started = True
                self.events.append(MixerEvent("started", v.id, v.effect_id, v.bus, v.owner, self.frames_rendered))
            acc = bus_out[v.bus]
            bus_out[v.bus] = block if acc is None else acc + block

        active_buses = {v.bus for v in self.voices if not v.done}
        alpha = min(1.0, n / max(1.0, self.duck_time * self.sample_rate))
        for b, cfg in self.buses.items():
            target = db_to_gain(cfg.duck_db) if any(d in active_buses for d in cfg.duck_under) else 1.0
            self.duck_env[b] += (target - self.duck_env[b]) * alpha
            if bus_out[b] is not None:
                out += bus_out[b] * (self.bus_gain[b] * self.duck_env[b])

        if self.master_gain != 1.0:
            out *= self.master_gain
        peak = float(np.abs(out).max()) if n else 0.0
        if peak > self.ceiling:
            out *= self.ceiling / peak
            self.limited_blocks += 1

        alive = []
        for v in self.voices:
            if v.done:
                self.events.append(MixerEvent("ended", v.id, v.effect_id, v.bus, v.owner, self.frames_rendered))
            else:
                alive.append(v)
        self.voices = alive
        self.playing = frozenset((v.id, v.effect_id, v.bus, v.owner) for v in alive)
        self.frames_rendered += n
        return out
