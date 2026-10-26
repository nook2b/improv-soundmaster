"""Сбои, которые не должны ронять шоу: запись, вывод, генератор, пул, режиссёр музыки."""

import asyncio
import sys
import threading
import time
import types
from concurrent.futures import Future
from dataclasses import replace

import numpy as np
import pytest

from soundmaster.clock import FakeClock
from soundmaster.journal import Journal
from soundmaster.music.generators import LyriaRealtimeGenerator, MusicGenerator
from soundmaster.registry import MusicStyle

SR = 8000


# ---- запись -------------------------------------------------------------------------------
def test_track_survives_write_error_and_close_does_not_hang(tmp_path, monkeypatch):
    from soundmaster.audio import recording

    class FullDisk:
        frames = 0

        def __init__(self, *a):
            pass

        def write(self, block):
            raise OSError(28, "No space left on device")

        def close(self):
            pass

    monkeypatch.setattr(recording, "WavStreamWriter", FullDisk)
    journal = Journal(None, FakeClock(0.0))
    tr = recording.Track("stage", tmp_path / "stage.wav", 2, SR, journal)
    block = np.zeros((256, 2), np.float32)
    for i in range(6000):  # больше, чем вмещает очередь: поток разбирает её и после ошибки
        tr.push(block, i * 0.03)
    t0 = time.monotonic()
    tr.close()
    assert time.monotonic() - t0 < 5
    assert not tr._thread.is_alive()
    assert tr.error and "No space" in tr.error
    assert any(r["kind"] == "track_error" for r in journal.records)


def test_track_close_with_stuck_writer_returns(tmp_path, monkeypatch):
    from soundmaster.audio import recording

    release = threading.Event()

    class Stuck:
        frames = 0

        def __init__(self, *a):
            pass

        def write(self, block):
            release.wait()

        def close(self):
            pass

    monkeypatch.setattr(recording, "WavStreamWriter", Stuck)
    tr = recording.Track("stage", tmp_path / "stage.wav", 2, SR, Journal(None, FakeClock(0.0)))
    block = np.zeros((16, 2), np.float32)
    for i in range(4100):
        tr.push(block, 0.0)
    assert tr.q.full()
    t0 = time.monotonic()
    tr.close(timeout=0.2)
    assert time.monotonic() - t0 < 2  # раньше q.put(None) ждал вечно
    release.set()
    tr._thread.join(timeout=2)
    assert not tr._thread.is_alive()


# ---- вывод --------------------------------------------------------------------------------
def test_output_callback_outputs_silence_on_render_error():
    from soundmaster.audio.devices import SoundDeviceOutput

    class BadMixer:
        sample_rate = SR

        def render(self, frames):
            raise ValueError("сломалось")

    out = SoundDeviceOutput(BadMixer(), FakeClock(0.0))
    buf = np.ones((64, 2), np.float32)
    ti = types.SimpleNamespace(outputBufferDacTime=0.0, currentTime=0.0)
    out._callback(buf, 64, ti, None)  # исключение не уходит в PortAudio
    assert not buf.any()
    assert out.render_errors == 1
    n, text = out.take_error()
    assert n == 1 and "ValueError" in text
    assert out.take_error() == (1, None)


# ---- режиссёр музыки ----------------------------------------------------------------------
class FakeGen(MusicGenerator):
    """Генератор под контролем теста: prebuffer — сколько кадров сразу положить в буфер."""

    name = "fake"

    def __init__(self, sr, prebuffer=0):
        super().__init__(sr)
        self.prebuffer = prebuffer
        self.starts = self.stops = 0

    def start(self, preset, buffer):
        self.starts += 1
        self.buffer = buffer
        if self.prebuffer:
            buffer.push(np.zeros((self.prebuffer, 2), np.float32))

    def update(self, preset, cut=False):
        pass

    def stop(self):
        self.stops += 1


def _stuck_loader(music):
    futs = []

    def submit(*a, **k):
        f = Future()
        futs.append((f, a))
        return f

    music._loader.submit = submit
    return futs


def _fake_pool(registry, genre):
    import soundfile

    for t in registry.pool[genre]:
        path = registry.sounds_dir / t.file
        path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(path), np.full((SR, 2), 0.05, np.float32), SR, format="FLAC")


def test_file_choice_cancels_pending_generator(system, registry):
    make, engine, mixer, journal = system
    music = make().music
    gen = FakeGen(SR)  # буфер не набирается — старт «висит»
    music.generator = gen
    music.set_generator_enabled(True)
    assert music.set_state("romantic", owner="ai") == "generator_pending"
    plain = replace(registry.music_states["tension"], id="plain", generator={})
    assert music.play_state(plain, "manual") == "file"
    assert "music" not in music._pending and gen.stops >= 1
    assert music.current().id == "plain"
    gen.buffer.push(np.zeros((2 * SR, 2), np.float32))  # запоздалый буфер старого старта
    music.tick()
    assert music.active["music"].state.id == "plain" and music.active["music"].source == "file"


def test_file_choice_cancels_pending_pool_load(system, registry):
    make, engine, mixer, journal = system
    music = make().music
    _fake_pool(registry, "folk")
    futs = _stuck_loader(music)
    assert music.play_pool("folk") == "pool_loading"
    plain = replace(registry.music_states["tension"], id="plain", generator={})
    music.play_state(plain, "manual")
    assert music._pool_pending is None and music.current().id == "plain"
    for f, a in futs:  # запись догрузилась — но выбор уже другой
        f.set_result(a[0](*a[1:]))
    music.tick()
    assert music.active["music"].state.id == "plain"


def test_generator_failure_falls_back_once_while_pool_loads(system, registry):
    make, engine, mixer, journal = system
    music = make().music
    _fake_pool(registry, "folk")
    gen = FakeGen(SR, prebuffer=2 * SR)
    music.generator = gen
    music.set_generator_enabled(True)
    style = MusicStyle("folk", "happy", "medium", "medium", "balalaika")
    music.set_style(style)
    assert music.channel_source("music") == "generator"
    _stuck_loader(music)
    pool_next = dict(music._pool_next)
    gen.error = "нет сети"
    for _ in range(50):
        music.tick()
    acts = [r.get("action") for r in journal.records]
    assert acts.count("fallback") == 1
    assert acts.count("pool_fallback") == 1 and acts.count("pool_loading") == 1
    assert music._pool_next == pool_next
    assert music.current().id == f"style:{style.id}"  # для модели выбор прежний


# ---- Lyria --------------------------------------------------------------------------------
def test_lyria_stop_does_not_block_and_closes_loop(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    gen = LyriaRealtimeGenerator(48000)
    seen = {"stopped": False, "cancelled": False}
    started = threading.Event()

    class Session:
        async def stop(self):
            await asyncio.sleep(0.3)  # медленная сеть
            seen["stopped"] = True

    async def main():
        gen._session = Session()
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise

    monkeypatch.setattr(gen, "_main", main)
    from soundmaster.audio.mixer import StreamBuffer

    gen.start({"prompts": []}, StreamBuffer(1.0, 48000))
    assert started.wait(2)
    loop, thread = gen._loop, gen._thread
    t0 = time.monotonic()
    gen.stop()
    assert time.monotonic() - t0 < 0.1  # пульт не ждёт сеть
    gen.stop()  # повторная остановка безопасна
    thread.join(timeout=3)
    assert not thread.is_alive() and loop.is_closed()
    assert seen == {"stopped": True, "cancelled": True}


# ---- пул ----------------------------------------------------------------------------------
def test_record_lyria_times_out_on_silent_stream(monkeypatch):
    from soundmaster.music import pool as mp

    class Session:
        async def set_weighted_prompts(self, prompts):
            pass

        async def play(self):
            pass

        async def receive(self):
            await asyncio.Event().wait()  # поток замолчал: ни одного сообщения
            yield None

    class Connect:
        async def __aenter__(self):
            return Session()

        async def __aexit__(self, *a):
            return False

    class Client:
        def __init__(self, **k):
            self.aio = types.SimpleNamespace(live=types.SimpleNamespace(
                music=types.SimpleNamespace(connect=lambda model: Connect())))

    gtypes = types.SimpleNamespace(WeightedPrompt=lambda **k: k, LiveMusicGenerationConfig=lambda **k: k)
    genai = types.ModuleType("google.genai")
    genai.Client, genai.types = Client, gtypes
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(mp.record_lyria({"prompts": [{"text": "x", "weight": 1.0}]}, 1.0, "k", timeout=0.3))
    assert time.monotonic() - t0 < 3


def test_make_pool_rejected_plan_once_and_write_error_is_per_track(registry, tmp_path, monkeypatch):
    from soundmaster.music import pool as mp

    monkeypatch.setattr(mp.shutil, "which", lambda name: "/usr/bin/" + name)
    reg = replace(registry, sounds_dir=tmp_path)
    tracks = sorted(reg.pool["underscore"], key=lambda t: t.n)
    assert len(tracks) >= 3
    rejected, broken = tracks[0], tracks[1]
    x = (np.random.default_rng(1).standard_normal((int(40 * mp.SR), 2)) * 0.05).astype(np.float32)
    calls: dict[int, int] = {}

    real_plan = mp.eleven_plan

    def plan_with_n(registry, t, seconds):
        return {**real_plan(registry, t, seconds), "_n": t.n}

    async def rec(plan, key):
        calls[plan["_n"]] = calls.get(plan["_n"], 0) + 1
        if plan["_n"] == rejected.n:
            raise mp.PlanRejected("похоже на название песни")
        return x

    def write_flac(path, data, sr=mp.SR):
        if path.name == broken.file.split("/")[-1]:
            raise OSError(28, "No space left on device")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"flac")

    monkeypatch.setattr(mp, "eleven_plan", plan_with_n)
    monkeypatch.setattr(mp, "write_flac", write_flac)
    rep = mp.make_pool(reg, seconds=3, parallel=1, only="underscore", progress=lambda s: None,
                       recorder=rec, takes=3, engine="elevenlabs")
    assert calls[rejected.n] == 1  # отклонённый план не повторяем в каждом дубле
    errs = dict(rep.errors)
    assert set(errs) == {rejected.file, broken.file} and "No space" in errs[broken.file]
    assert len(rep.made) == len(tracks) - 2  # сбой записи одной мелодии не остановил остальные
