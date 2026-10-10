"""Музыкальный режиссёр: включает «музыкальное состояние» из файла или генератора.

Источник — генератор (если включён и исправен) или библиотечный файл того же состояния.
При сбое генератора или опустевшем буфере — плавный переход на файл (или тишину), а не обрыв.
Стиль из словаря (жанр/настроение/энергия/темп) — то же состояние, собранное на лету.
Если звучать нечем, состояние всё равно считается активным (source="none"): иначе модель
повторяла бы тот же выбор, а пульт показывал бы, что музыки нет, хотя она «выбрана».

Пул записанных мелодий (registry.pool): ручные кнопки жанров (повторное нажатие — следующая
мелодия, плавный переход) и резерв при сбое генератора — мелодия того же жанра. Файлы читаются
с диска в фоне, в памяти — только несколько последних.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from ..audio.mixer import StreamBuffer
from ..audio.wavio import load_sound
from ..engine import Engine
from ..journal import Journal
from ..registry import MusicState, MusicStyle, PoolTrack, Registry
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
        pool_cache: int = 6,
        generator_retry: float = 60.0,
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
        self._pool_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._pool_cache_size = pool_cache
        self._pool_loading: dict[str, Future] = {}
        self._pool_pending: tuple[MusicState, str] | None = None  # ждёт загрузки файла
        self._pool_next: dict[str, int] = {}
        self._loader = ThreadPoolExecutor(1, thread_name_prefix="pool-load")
        self.generator_retry = generator_retry
        self._gen_failed_at: float | None = None

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
                self._pool_pending = None
                buf = StreamBuffer(self.buffer_seconds, self.sample_rate)
                self._stop_generator()
                self.generator.start(state.generator, buf)
                self._pending[state.channel] = (state, owner, buf)
                self.journal.write("music", action="generator_start", state=state_id, owner=owner,
                                   generator=self.generator.name, preset=state.generator)
                self.tick()
                return "generator_pending"
            self.journal.write("music", action="generator_unhealthy", state=state_id, error=self.generator.error)
            self._maybe_retry_generator()

        return self._fallback(state, owner)

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

    # ---- пул записанных мелодий -----------------------------------------------------------
    def pool_tracks(self, genre: str) -> list[PoolTrack]:
        """Записанные мелодии жанра, у которых есть файл."""
        return [t for t in self.registry.pool.get(genre, []) if (self.registry.sounds_dir / t.file).exists()]

    def pool_state(self, t: PoolTrack, of: int) -> MusicState:
        v = self.registry.vocabulary
        assert v is not None
        st = t.style
        label = (f"{v.genres[st.genre].label} {t.n}/{of} · {v.instruments[st.instrument].label}, "
                 f"{v.moods[st.mood].label}")
        return MusicState(id=f"pool:{st.genre}/{t.n}", channel="music", label=label, file=t.file,
                          fade_in=3.0, fade_out=3.0, gain_db=v.gain_db)

    def play_pool(self, genre: str, owner: str = "manual") -> str:
        """Следующая мелодия жанра (по кругу) с плавным переходом. Генератор при этом молчит."""
        tracks = self.pool_tracks(genre)
        if not tracks:
            self.journal.write("music", action="pool_empty", genre=genre)
            return "no_pool"
        i = self._pool_next.get(genre, 0) % len(tracks)
        self._pool_next[genre] = i + 1
        result = self._play_pool_track(self.pool_state(tracks[i], len(self.registry.pool[genre])), owner)
        self._prefetch(tracks[(i + 1) % len(tracks)].file)
        return result

    def _pool_for(self, state: MusicState) -> PoolTrack | None:
        """Резерв из пула для стиля: тот же жанр (лучше тот же инструмент, потом настроение)."""
        if not state.id.startswith("style:"):
            return None
        style = MusicStyle.parse(state.id.removeprefix("style:"))
        tracks = self.pool_tracks(style.genre) or self.pool_tracks("underscore")
        if not tracks:
            return None
        for same in (lambda t: t.style.instrument == style.instrument, lambda t: t.style.mood == style.mood):
            hit = [t for t in tracks if same(t)]
            if hit:
                return hit[0]
        i = self._pool_next.get(style.genre, 0) % len(tracks)
        self._pool_next[style.genre] = i + 1
        return tracks[i]

    def _fallback(self, state: MusicState, owner: str) -> str:
        """Генератор недоступен: запись из пула (id стиля сохраняется — модель не будет её «перевыбирать»)."""
        t = self._pool_for(state)
        if t is None:
            return self._start_file(state, owner)
        st = replace(state, file=t.file, fade_in=3.0, fade_out=3.0)
        self.journal.write("music", action="pool_fallback", state=state.id, file=t.file)
        return self._play_pool_track(st, owner)

    def _play_pool_track(self, state: MusicState, owner: str) -> str:
        data = self._pool_cache.get(state.file or "")
        if data is None:
            self._prefetch(state.file or "")
            self._pool_pending = (state, owner)
            self.journal.write("music", action="pool_loading", state=state.id, file=state.file)
            return "pool_loading"
        self._pool_pending = None
        self._pool_cache.move_to_end(state.file or "")
        self.engine.sounds[state.file or ""] = data
        if self.generator and (self.channel_source("music") == "generator" or "music" in self._pending):
            self._pending.pop("music", None)
            self._stop_generator()
        self._dip_until = None
        self._start_file(state, owner)
        return "pool"

    def channel_source(self, channel: str) -> str | None:
        cur = self.active.get(channel)
        return cur.source if cur else None

    def _prefetch(self, rel: str) -> None:
        if not rel or rel in self._pool_cache or rel in self._pool_loading:
            return
        path = self.registry.sounds_dir / rel
        self._pool_loading[rel] = self._loader.submit(load_sound, path, self.sample_rate)

    def _pool_tick(self) -> None:
        for rel, fut in list(self._pool_loading.items()):
            if not fut.done():
                continue
            del self._pool_loading[rel]
            try:
                self._pool_cache[rel] = fut.result()
            except Exception as e:  # noqa: BLE001 — битый файл: пульт и автоматика работают дальше
                self.journal.write("music", action="pool_error", file=rel, error=f"{type(e).__name__}: {e}")
                if self._pool_pending and self._pool_pending[0].file == rel:
                    st, owner = self._pool_pending
                    self._pool_pending = None
                    self._start_file(replace(st, file=None), owner)
                continue
            while len(self._pool_cache) > self._pool_cache_size:
                old, _ = self._pool_cache.popitem(last=False)
                cur = self.active.get("music")
                if cur and cur.state.file == old:  # играющий не выгружаем
                    self._pool_cache[old] = self.engine.sounds[old]
                    self._pool_cache.move_to_end(old, last=False)
                    break
                self.engine.sounds.pop(old, None)
        if self._pool_pending and (self._pool_pending[0].file or "") in self._pool_cache:
            st, owner = self._pool_pending
            self._play_pool_track(st, owner)

    def _maybe_retry_generator(self) -> None:
        """Генератор упал (например, не было интернета) — через generator_retry с пробуем снова."""
        if self.generator is None:
            return
        now = self.engine.clock.now()
        if self._gen_failed_at is None:
            self._gen_failed_at = now
        elif now - self._gen_failed_at >= self.generator_retry:
            self._gen_failed_at = None
            self.generator.error = None
            self.journal.write("music", action="generator_retry")

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
            self._pool_pending = None
        pending = self._pending.pop(channel, None)
        cur = self.active.get(channel)
        if pending or (cur and cur.source == "generator"):
            self._stop_generator()
        self._fade_out_current(channel)
        if cur or pending:
            self.journal.write("music", action="stop", channel=channel)

    def current(self, channel: str = "music") -> MusicState | None:
        """Что выбрано на канале: играющее, ждущее предзаполнения буфера или загрузки записи."""
        if channel in self._pending:
            return self._pending[channel][0]
        if self._pool_pending and self._pool_pending[0].channel == channel:
            return self._pool_pending[0]
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
        self._pool_tick()
        if self._dip_until is not None and self.engine.clock.now() >= self._dip_until:
            self._dip_until = None
            cur = self.active.get("music")
            if cur and cur.voice_id is not None:
                self.engine.mixer.fade(cur.voice_id, 1.0, self.cut_rise)
        for channel, (state, owner, buf) in list(self._pending.items()):
            if self.generator and not self.generator.healthy:
                del self._pending[channel]
                self.journal.write("music", action="fallback", state=state.id, reason=self.generator.error)
                self._maybe_retry_generator()
                self._fallback(state, owner)
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
                if failed:
                    self._maybe_retry_generator()
                self._fallback(cur.state, cur.owner)

    def status(self) -> dict:
        return {
            "generator": self.generator.name if self.generator else None,
            "generator_enabled": self.use_generator,
            "generator_error": self.generator.error if self.generator else None,
            "active": {ch: {"state": a.state.id, "label": a.state.label, "source": a.source, "owner": a.owner}
                       for ch, a in self.active.items()},
            "pending": {ch: s.id for ch, (s, _, _) in self._pending.items()},
            "pool_loading": self._pool_pending[0].id if self._pool_pending else None,
        }


def _loop_source(data):
    from ..audio.mixer import BufferSource

    return BufferSource(data, loop=True)
