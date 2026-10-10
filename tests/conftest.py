from pathlib import Path

import numpy as np
import pytest

from soundmaster.audio.mixer import Mixer
from soundmaster.clock import FakeClock
from soundmaster.controller import Controller
from soundmaster.engine import Engine
from soundmaster.journal import Journal
from soundmaster.music.director import MusicDirector
from soundmaster.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent
SR = 8000


@pytest.fixture
def registry(tmp_path):
    # Своя пустая папка звуков: тесты не зависят от того, что лежит в sounds/ (пул, импорт).
    reg = load_registry(ROOT / "config" / "registry.yaml")
    reg.sounds_dir = tmp_path / "sounds"
    return reg


@pytest.fixture
def clock():
    return FakeClock(100.0)


@pytest.fixture
def system(registry, clock):
    journal = Journal(None, clock)
    mixer = Mixer(sample_rate=SR)
    engine = Engine(registry, mixer, clock, journal)
    # Вместо файлов — короткие синтетические буферы, чтобы тесты не зависели от диска.
    for e in registry.effects.values():
        engine.sounds[e.file] = np.full((SR // 2, 2), 0.1, np.float32)
    for s in registry.music_states.values():
        engine.sounds[s.file] = np.full((SR, 2), 0.05, np.float32)
    music = MusicDirector(registry, engine, journal)

    def make(clean: bool = False) -> Controller:
        return Controller(registry, engine, music, journal, clock, clean_actor_feed=clean)

    return make, engine, mixer, journal
