"""Музыкальный режиссёр: включает «музыкальное состояние» из файла или генератора.

Источник — генератор (если включён и исправен) или библиотечный файл того же состояния.
При сбое генератора или опустевшем буфере — плавный переход на файл (или тишину), а не обрыв.
Стиль из словаря (жанр/настроение/энергия/темп) — то же состояние, собранное на лету.
Если звучать нечем, состояние всё равно считается активным (source="none"): иначе модель
повторяла бы тот же выбор, а пульт показывал бы, что музыки нет, хотя она «выбрана».
"""

from __future__ import annotations

from dataclasses import dataclass

from ..audio.mixer import StreamBuffer
from ..engine import Engine
from ..journal import Journal
from ..registry import MusicState, MusicStyle, Registry
from .generators import MusicGenerator


@dataclass
class ActiveMusic:
    state: MusicState
    source: str  # "generator" | "file" | "none"
    voice_id: int | None
    owner: str
    buffer: StreamBuffer | None = None
    underruns_seen: int = 0
    bridges_seen: int = 0


class MusicDirector:
    def __init__(
        self,
        registry: Registry,
        engine: Engine,
        journal: Journal,
        generator: MusicGenerator | None = None,
        prebuffer_seconds: float = 1.0,
        buffer_seconds: float = 8.0,
        max_underruns: int = 20,
        cut_dip_db: float = -18.0,
        cut_dip_fade: float = 1.5,
        cut_dip_hold: float = 3.0,
        cut_rise: float = 3.0,
    ):
        self.registry = registry
        self.engine = engine
        self.journal = journal
        self.generator = generator
        self.use_generator = generator is not None
        self.prebuffer = prebuffer_seconds
        self.buffer_seconds = buffer_seconds
        self.max_underruns = max_underruns
        self.active: dict[str, ActiveMusic] = {}  # канал → что играет
        self._pending: dict[str, tuple[MusicState, str, StreamBuffer]] = {}  # ждём предзаполнения буфера
        # «Монтаж» при играющем генераторе: музыка проваливается, генератор сразу (без плавного
        # перехода) переключается, потом новая музыка поднимается — смена сцены слышна.
        self.cut_dip_gain = 10 ** (cut_dip_db / 20)
        self.cut_dip_fade = cut_dip_fade
        self.cut_dip_hold = cut_dip_hold
        self.cut_rise = cut_rise
        self._dip_until: float | None = None

    @property
    def sample_rate(self) -> int:
        return self.engine.mixer.sample_rate

    def set_state(self, state_id: str, owner: str = "manual") -> str:
        state = self.registry.music_states.get(state_id)
        if state is None:
            raise KeyError(f"неизвестное музыкальное состояние: {state_id}")
        return self.play_state(state, owner)

    def set_style(self, style: MusicStyle, owner: str = "ai", cut: bool = False) -> str:
        return self.play_state(self.registry.music_style_state(style), owner, cut)

    def play_state(self, state: MusicState, owner: str = "manual", cut: bool = False) -> str:
        """cut — смена сцены: при играющем генераторе провал громкости и резкая смена вместо перехода."""
        state_id = state.id
        cur = self.active.get(state.channel)
        chosen = self.current(state.channel)
        if chosen and chosen.id == state_id:
            return "already_playing"

        # Генератор один и обслуживает только шину music; атмосфера — из файлов.
        if self.use_generator and self.generator and state.generator and state.channel == "music":
            if self.generator.healthy and cur and cur.source == "generator":
                # Генератор уже играет: меняем состояние без остановки потока.
                now = self.engine.clock.now()
                if cut and cur.voice_id is not None:
                    self.engine.mixer.fade(cur.voice_id, self.cut_dip_gain, self.cut_dip_fade)
                    self._dip_until = now + self.cut_dip_hold
                elif self._dip_until is not None:
                    cut = True  # выбор модели, пока громкость в провале, — тоже сразу
                    self._dip_until = max(self._dip_until, now + 1.5)
                self.generator.update(state.generator, cut=cut)
                cur.state = state
                cur.owner = owner
                self.journal.write("music", action="generator_cut" if cut else "generator_update",
                                   state=state_id, owner=owner, preset=state.generator)
                return "generator_update"
            if self.generator.healthy and state.channel in self._pending:
                # Генератор ещё набирает буфер: меняем пресет, не перезапуская поток.
                _, _, buf = self._pending[state.channel]
                self.generator.update(state.generator)
                self._pending[state.channel] = (state, owner, buf)
                self.journal.write("music", action="generator_update", state=state_id, owner=owner,
                                   preset=state.generator)
                return "generator_pending"
            if self.generator.healthy:
                buf = StreamBuffer(self.buffer_seconds, self.sample_rate)
                self._stop_generator()
                self.generator.start(state.generator, buf)
                self._pending[state.channel] = (state, owner, buf)
                self.journal.write("music", action="generator_start", state=state_id, owner=owner,
                                   generator=self.generator.name, preset=state.generator)
                self.tick()
                return "generator_pending"
            self.journal.write("music", action="generator_unhealthy", state=state_id, error=self.generator.error)

        return self._start_file(state, owner)

    def _start_file(self, state: MusicState, owner: str) -> str:
        if not state.file or state.file not in self.engine.sounds:
            self.stop(state.channel)
            self.active[state.channel] = ActiveMusic(state, "none", None, owner)
            self.journal.write("music", action="no_source", state=state.id, owner=owner, preset=state.generator)
            return "no_source"
        self._fade_out_current(state.channel)
        data = self.engine.sounds[state.file]
        vid = self.engine.play_buffer(f"music:{state.id}", _loop_source(data), state.channel, owner,
                                      state.gain_db, state.fade_in, {"music_state": state.id, "source": "file"})
        self.active[state.channel] = ActiveMusic(state, "file", vid, owner)
        self.journal.write("music", action="file_start", state=state.id, owner=owner, voice_id=vid)
        return "file"

    def _fade_out_current(self, channel: str) -> None:
        cur = self.active.pop(channel, None)
        if cur and cur.voice_id is not None:
            self.engine.mixer.stop(cur.voice_id, fade=cur.state.fade_out)

    def _stop_generator(self) -> None:
        if self.generator:
            self.generator.stop()

    def stop(self, channel: str = "music") -> None:
        if channel == "music":
            self._dip_until = None
        pending = self._pending.pop(channel, None)
        cur = self.active.get(channel)
        if pending or (cur and cur.source == "generator"):
            self._stop_generator()
        self._fade_out_current(channel)
        if cur or pending:
            self.journal.write("music", action="stop", channel=channel)

    def current(self, channel: str = "music") -> MusicState | None:
        """Что выбрано на канале: играющее или ждущее предзаполнения буфера."""
        if channel in self._pending:
            return self._pending[channel][0]
        cur = self.active.get(channel)
        return cur.state if cur else None

    def stop_all(self) -> None:
        for ch in list(self.active) + list(self._pending):
            self.stop(ch)

    def set_generator_enabled(self, enabled: bool) -> None:
        self.use_generator = enabled and self.generator is not None
        self.journal.write("music", action="generator_enabled", enabled=self.use_generator)

    def tick(self) -> None:
        """Вызывать периодически из основного цикла."""
        if self._dip_until is not None and self.engine.clock.now() >= self._dip_until:
            self._dip_until = None
            cur = self.active.get("music")
            if cur and cur.voice_id is not None:
                self.engine.mixer.fade(cur.voice_id, 1.0, self.cut_rise)
        for channel, (state, owner, buf) in list(self._pending.items()):
            if self.generator and not self.generator.healthy:
                del self._pending[channel]
                self.journal.write("music", action="fallback", state=state.id, reason=self.generator.error)
                self._start_file(state, owner)
            elif buf.frames >= self.prebuffer * self.sample_rate:
                del self._pending[channel]
                self._fade_out_current(channel)
                vid = self.engine.play_buffer(f"music:{state.id}", buf, channel, owner, state.gain_db,
                                              state.fade_in, {"music_state": state.id, "source": "generator"})
                self.active[channel] = ActiveMusic(state, "generator", vid, owner, buf, buf.underruns)
                self.journal.write("music", action="generator_playing", state=state.id, voice_id=vid)

        for channel, cur in list(self.active.items()):
            if cur.source != "generator" or cur.buffer is None:
                continue
            if cur.buffer.bridges > cur.bridges_seen:
                cur.bridges_seen = cur.buffer.bridges
                self.journal.write("music", action="generator_bridge", state=cur.state.id,
                                   reason="поток прервался — повторяем последние секунды, пока генератор переподключается")
            new_underruns = cur.buffer.underruns - cur.underruns_seen
            failed = self.generator is not None and not self.generator.healthy
            if failed or new_underruns > self.max_underruns:
                reason = self.generator.error if failed else f"буфер пуст ({new_underruns} раз)"
                self.journal.write("music", action="fallback", state=cur.state.id, reason=reason)
                self._stop_generator()
                self._dip_until = None
                self._start_file(cur.state, cur.owner)

    def status(self) -> dict:
        return {
            "generator": self.generator.name if self.generator else None,
            "generator_enabled": self.use_generator,
            "generator_error": self.generator.error if self.generator else None,
            "active": {ch: {"state": a.state.id, "label": a.state.label, "source": a.source}
                       for ch, a in self.active.items()},
            "pending": {ch: s.id for ch, (s, _, _) in self._pending.items()},
        }


def _loop_source(data):
    from ..audio.mixer import BufferSource

    return BufferSource(data, loop=True)
