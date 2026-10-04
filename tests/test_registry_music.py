import time

import numpy as np
import pytest

from soundmaster.music.generators import MusicGenerator, TestToneGenerator
from soundmaster.registry import RegistryError, load_registry

SR = 8000


def test_registry_loads(registry):
    assert registry.effect("gunshot").human_only
    assert registry.scene_end_effect == "scene_end"
    assert "romantic" in registry.music_states


def test_loud_auto_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("effects:\n  boom: {file: a.wav, risk: loud, mode: AUTO}\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(p)


def test_reserved_hotkey_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("effects:\n  x: {file: a.wav, hotkey: n}\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(p)


def test_duplicate_hotkey_rejected(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("effects:\n  a: {file: a.wav, hotkey: '1'}\n  b: {file: b.wav, hotkey: '1'}\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(p)


def test_music_from_file_without_generator(system):
    make, engine, mixer, _ = system
    c = make()
    assert c.manual_music("romantic") == "file"
    assert c.manual_music("romantic") == "already_playing"


class BrokenGenerator(MusicGenerator):
    name = "broken"

    def start(self, preset, buffer):
        self.error = "нет сети"

    def update(self, preset):
        pass

    def stop(self):
        pass


def test_generator_failure_falls_back_to_file(system):
    make, engine, mixer, journal = system
    c = make()
    c.music.generator = BrokenGenerator(SR)
    c.music.set_generator_enabled(True)
    result = c.manual_music("romantic")
    c.music.tick()
    assert result in ("generator_pending", "file")
    assert c.music.active["music"].source == "file"
    assert any(r.get("action") == "fallback" for r in journal.records)


def test_test_tone_generator_streams_and_updates(system):
    make, engine, mixer, _ = system
    c = make()
    gen = TestToneGenerator(SR, block=256, ahead_seconds=1.5)
    c.music.generator = gen
    c.music.set_generator_enabled(True)
    assert c.manual_music("romantic") == "generator_pending"
    deadline = time.time() + 3
    while "music" not in c.music.active and time.time() < deadline:
        c.music.tick()
        time.sleep(0.02)
    assert c.music.active["music"].source == "generator"
    out = mixer.render(SR // 4)
    assert np.abs(out).max() > 0
    assert c.manual_music("tension") == "generator_update"
    assert gen.preset["root_hz"] == 110
    c.manual_music(None)
    gen.stop()


def test_vocabulary_builds_generator_preset(registry):
    from soundmaster.registry import MusicStyle

    st = registry.music_style_state(MusicStyle("noir", "tense", "low", "slow"))
    assert st.channel == "music" and st.file == registry.music_states["tension"].file
    assert [p["text"] for p in st.generator["prompts"]][0].startswith("film noir")
    assert "bpm" not in st.generator["config"] and st.generator["config"]["density"] == 0.2
    assert any(p["text"] == "slow tempo" for p in st.generator["prompts"])  # темп словами — плавная смена
    assert registry.vocabulary.validate(MusicStyle("noir", "x", "low", "slow")) == "unknown_mood"


def test_vocabulary_fallback_must_exist(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("music_vocabulary:\n  genres: {a: {}}\n  moods: {b: {fallback: nope}}\n"
                 "  energy: {c: {}}\n  tempo: {d: {}}\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(p)


def test_style_without_any_source_is_still_tracked(system):
    """Нет генератора и нет файла настроения — тишина, но выбор помнится (без повторов от модели)."""
    from soundmaster.registry import MusicStyle

    make, engine, mixer, journal = system
    c = make()
    s = MusicStyle("comedy", "happy", "medium", "fast")  # у «радости» нет резервного файла
    assert c.music.set_style(s) == "no_source"
    assert c.music.current("music").id == f"style:{s.id}"
    assert c.music.set_style(s) == "already_playing"
    c.music.stop("music")
    assert c.music.current("music") is None
