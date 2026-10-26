"""Генераторы потоковой музыки.

Генератор получает проверенный пресет из реестра (а не свободный текст от модели) и пишет
стерео-блоки float32 в `StreamBuffer`. Микшер читает буфер как обычный голос на шине music.

- `TestToneGenerator` — локальный синтезатор аккордов для проверки цепочки без сети и ключей.
- `LyriaRealtimeGenerator` — адаптер Google Lyria RealTime (экспериментальная модель Google;
  поток проверен на реальном ключе: первый звук через ~3 с; смена стиля — плавный переход;
  включается в config/show.yaml).
"""

from __future__ import annotations

import asyncio
import os
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..audio.mixer import StreamBuffer


class MusicGenerator(ABC):
    name = "base"

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.buffer: StreamBuffer | None = None
        self.error: str | None = None

    @property
    def healthy(self) -> bool:
        return self.error is None

    @abstractmethod
    def start(self, preset: dict[str, Any], buffer: StreamBuffer) -> None: ...

    @abstractmethod
    def update(self, preset: dict[str, Any], cut: bool = False) -> None:
        """Сменить состояние без остановки потока; cut — сразу, без плавного перехода («монтаж»)."""

    @abstractmethod
    def stop(self) -> None: ...


class TestToneGenerator(MusicGenerator):
    """Мягкий аккорд с пульсацией. Пресет: root_hz, bpm, chord (полутоны), brightness 0..1."""

    __test__ = False  # не тестовый класс pytest
    name = "test_tone"

    def __init__(self, sample_rate: int, block: int = 2048, ahead_seconds: float = 2.0):
        super().__init__(sample_rate)
        self.block = block
        self.ahead = int(ahead_seconds * sample_rate)
        self.preset: dict[str, Any] = {}
        self._phase = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, preset: dict[str, Any], buffer: StreamBuffer) -> None:
        self.preset = dict(preset)
        self.buffer = buffer
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="test-tone-gen", daemon=True)
        self._thread.start()

    def update(self, preset: dict[str, Any], cut: bool = False) -> None:
        self.preset = dict(preset)

    def stop(self) -> None:
        self._stop.set()
        if self.buffer:
            self.buffer.closed = True
        if self._thread:
            self._thread.join(timeout=1)

    def render_block(self, n: int) -> np.ndarray:
        p = self.preset
        root = float(p.get("root_hz", 220.0))
        bpm = float(p.get("bpm", 70.0))
        chord = p.get("chord", [0, 4, 7])
        bright = float(p.get("brightness", 0.3))
        t = (self._phase + np.arange(n)) / self.sample_rate
        sig = np.zeros(n, np.float64)
        for semis in chord:
            f = root * 2 ** (semis / 12)
            sig += np.sin(2 * np.pi * f * t) + bright * 0.3 * np.sin(2 * np.pi * 2 * f * t)
        beat = 0.6 + 0.4 * np.cos(2 * np.pi * (bpm / 60.0) * t) ** 2
        sig = 0.12 * sig / max(1, len(chord)) * beat
        self._phase += n
        return np.stack([sig, sig], axis=1).astype(np.float32)

    def _run(self) -> None:
        import time

        assert self.buffer is not None
        while not self._stop.is_set():
            if self.buffer.frames < self.ahead:
                self.buffer.push(self.render_block(self.block))
            else:
                time.sleep(self.block / self.sample_rate / 2)


def blend_prompts(old: list[dict[str, Any]], new: list[dict[str, Any]], a: float) -> list[dict[str, Any]]:
    """Смесь описаний для плавного перехода: a=0 — старое, a=1 — новое (веса масштабируются).

    Одинаковые тексты складываются; описания с почти нулевым весом не отправляются.
    """
    out: dict[str, float] = {}
    for prompts, k in ((old, 1.0 - a), (new, a)):
        for p in prompts:
            out[p["text"]] = out.get(p["text"], 0.0) + float(p.get("weight", 1.0)) * k
    return [{"text": t, "weight": round(w, 3)} for t, w in out.items() if w >= 0.05]


class LyriaRealtimeGenerator(MusicGenerator):
    """Google Lyria RealTime через пакет google-genai (v1alpha). Экспериментально.

    Пресет: prompts — список {text, weight}; config — density, brightness, guidance и т. п.
    Ключ — переменная окружения GEMINI_API_KEY (в репозиторий и журнал не попадает).
    Поток: 48 кГц, 16 бит, стерео. Смена состояния — плавная: описания смешиваются шагами за
    `transition_s` секунд (темп в словаре задан словами, а не bpm: смена bpm требует сброса
    контекста и звучит обрывом). При обрыве сессии — до `reconnects` переподключений подряд;
    если не вышло, генератор помечается неисправным, и MusicDirector переходит на файл.
    """

    name = "lyria_realtime"
    MODEL = "models/lyria-realtime-exp"

    def __init__(self, sample_rate: int, transition_s: float = 8.0, steps: int = 4, reconnects: int = 3):
        super().__init__(sample_rate)
        if sample_rate != 48000:
            raise ValueError("Lyria RealTime отдаёт 48 кГц — поставьте sample_rate: 48000")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session = None
        self._preset: dict[str, Any] = {}
        self.transition_s = transition_s
        self.steps = max(1, steps)
        self.reconnects = reconnects
        self._ramp: asyncio.Task | None = None
        self._stopping = False

    def start(self, preset: dict[str, Any], buffer: StreamBuffer) -> None:
        if not os.environ.get("GEMINI_API_KEY"):
            self.error = "нет GEMINI_API_KEY"
            return
        self.buffer = buffer
        self._preset = dict(preset)
        self._stopping = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, args=(self._loop,), name="lyria", daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._main(), self._loop)

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            loop.close()

    async def _set_prompts(self, prompts: list[dict[str, Any]]) -> None:
        from google.genai import types  # type: ignore

        await self._session.set_weighted_prompts(
            prompts=[types.WeightedPrompt(text=p["text"], weight=float(p.get("weight", 1.0))) for p in prompts])

    async def _apply(self, preset: dict[str, Any], previous: dict[str, Any] | None = None,
                     cut: bool = False) -> None:
        from google.genai import types  # type: ignore

        new_prompts = preset.get("prompts", [])
        if previous and previous.get("prompts") and self.transition_s > 0 and not cut:
            # Плавно: несколько промежуточных смесей старого и нового описания.
            for i in range(1, self.steps + 1):
                await self._set_prompts(blend_prompts(previous["prompts"], new_prompts, i / self.steps))
                if i < self.steps:
                    await asyncio.sleep(self.transition_s / self.steps)
        else:
            await self._set_prompts(new_prompts)
        if preset.get("config"):
            await self._session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(**preset["config"]))
            # bpm и лад Lyria применяет только после сброса контекста (слышно как обрыв) —
            # сбрасываем, только если они заданы явно и изменились.
            old = (previous or {}).get("config") or {}
            new = preset["config"]
            if previous is not None and any(k in new and old.get(k) != new.get(k) for k in ("bpm", "scale")):
                await self._session.reset_context()
                return
        if cut:
            # «Монтаж»: громкость в провале, рывок сброса не слышен — новая музыка без хвоста старой.
            await self._session.reset_context()

    async def _main(self) -> None:
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1alpha"})
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            return
        failures = 0
        while not self._stopping:
            try:
                session = None
                async with client.aio.live.music.connect(model=self.MODEL) as session:
                    self._session = session
                    await self._apply(self._preset)
                    await session.play()
                    async for msg in session.receive():
                        chunks = getattr(getattr(msg, "server_content", None), "audio_chunks", None) or []
                        for ch in chunks:
                            failures = 0
                            pcm = np.frombuffer(ch.data, "<i2").astype(np.float32) / 32768.0
                            self.buffer.push(pcm.reshape(-1, 2))
                last = "сессия завершена"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            finally:
                if self._session is session:  # после перезапуска сессия уже может быть новой
                    self._session = None
            if self._stopping:
                return
            failures += 1
            if failures > self.reconnects:
                self.error = last
                return
            await asyncio.sleep(min(4.0, 0.5 * 2 ** failures))  # переподключение через 1, 2, 4 с

    def update(self, preset: dict[str, Any], cut: bool = False) -> None:
        previous, self._preset = self._preset, dict(preset)
        if self._session and self._loop:
            self._loop.call_soon_threadsafe(self._start_ramp, self._preset, previous, cut)

    def _start_ramp(self, preset: dict[str, Any], previous: dict[str, Any], cut: bool = False) -> None:
        # Новый переход отменяет незаконченный.
        if self._ramp and not self._ramp.done():
            self._ramp.cancel()
        self._ramp = asyncio.ensure_future(self._apply(preset, previous, cut))

    def stop(self) -> None:
        """Не блокирует вызывающий (asyncio-цикл пульта): остановка идёт в цикле Lyria."""
        self._stopping = True
        if self.buffer:
            self.buffer.closed = True
        loop, session = self._loop, self._session
        self._loop, self._session, self._ramp = None, None, None
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(session), loop)
        except RuntimeError:  # цикл уже закрыт
            pass

    @staticmethod
    async def _shutdown(session) -> None:
        """В цикле Lyria: остановить поток, отменить _main и переход (закрывает websocket), остановить цикл."""
        if session is not None:
            try:
                await asyncio.wait_for(session.stop(), 1.0)
            except BaseException:  # noqa: BLE001 — при остановке ошибка сети не важна
                pass
        me = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not me]
        for t in tasks:
            t.cancel()
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3.0)
            except BaseException:  # noqa: BLE001 — не закрылось вовремя: цикл всё равно останавливаем
                pass
        asyncio.get_running_loop().stop()


def make_generator(name: str | None, sample_rate: int) -> MusicGenerator | None:
    if not name or name == "none":
        return None
    if name == "test_tone":
        return TestToneGenerator(sample_rate)
    if name == "lyria_realtime":
        return LyriaRealtimeGenerator(sample_rate)
    raise ValueError(f"неизвестный генератор музыки: {name}")
