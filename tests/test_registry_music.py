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

    def update(self, preset, cut=False):
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

    st = registry.music_style_state(MusicStyle("noir", "tense", "low", "slow", "saxophone"))
    assert st.channel == "music" and st.file == registry.music_states["tension"].file
    assert [p["text"] for p in st.generator["prompts"]][0].startswith("film noir")
    assert st.generator["config"]["density"] == registry.vocabulary.energy["low"].config["density"]
    assert any("slow" in p["text"] for p in st.generator["prompts"])  # характер темпа словами, сам bpm — общий
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
    s = MusicStyle("comedy", "happy", "medium", "fast", "ukulele")  # у «радости» нет резервного файла
    assert c.music.set_style(s) == "no_source"
    assert c.music.current("music").id == f"style:{s.id}"
    assert c.music.set_style(s) == "already_playing"
    c.music.stop("music")
    assert c.music.current("music") is None


def test_import_sounds_converts_by_id_and_trims_silence(tmp_path):
    import shutil
    import subprocess

    import numpy as np

    from soundmaster.audio.wavio import load_sound, write_wav
    from soundmaster.importsounds import import_sounds

    if not shutil.which("ffmpeg"):
        pytest.skip("нужен ffmpeg")
    reg_path = tmp_path / "r.yaml"
    reg_path.write_text(
        "sounds_dir: s\neffects:\n  doorbell: {file: sfx/doorbell.wav}\n"
        "music_states:\n  rain: {file: ambience/rain.wav, channel: ambience}\n", encoding="utf-8")
    reg = load_registry(reg_path)
    src = tmp_path / "in"
    src.mkdir()
    sr = 44100
    t = np.arange(sr) / sr
    bell = np.concatenate([np.zeros(sr), 0.05 * np.sin(2 * np.pi * 880 * t)])  # 1 с тишины, потом звук
    write_wav(tmp_path / "bell.wav", np.stack([bell, bell], 1).astype(np.float32), sr)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_path / "bell.wav"), str(src / "Doorbell.mp3")],
                   check=True)
    write_wav(src / "rain.wav", np.stack([bell, bell], 1).astype(np.float32), sr)
    write_wav(src / "kazoo.wav", np.zeros((100, 2), np.float32), sr)

    rep = import_sounds(reg, src)
    assert sorted(sid for sid, _ in rep.imported) == ["doorbell", "rain"]
    assert [p.name for p in rep.unknown] == ["kazoo.wav"] and not rep.errors and not rep.missing
    data = load_sound(reg.path_of("sfx/doorbell.wav"), 48000)
    assert data.shape[1] == 2 and len(data) < 1.3 * 48000  # тишина в начале срезана
    assert np.abs(data[: 48000 // 20]).max() > 0.01  # звук начинается сразу
    assert import_sounds(reg, src).skipped  # повторно без --force — не трогаем
    with pytest.raises(FileNotFoundError):
        import_sounds(reg, tmp_path / "нет такой папки")


class RecordingGenerator(MusicGenerator):
    """Сразу заполняет буфер и запоминает смены пресета."""

    name = "rec"

    def start(self, preset, buffer):
        self.buffer = buffer
        buffer.push(np.zeros((2 * SR, 2), np.float32))
        self.updates = []

    def update(self, preset, cut=False):
        self.updates.append(cut)

    def stop(self):
        pass


def test_scene_cut_with_generator_dips_switches_at_once_then_rises(system, clock):
    from soundmaster.registry import MusicStyle

    make, engine, mixer, journal = system
    c = make()
    c.music_mode, c.scene_cut_mode, c.scene_cut_music = "AUTO", "AUTO", "neutral"
    c.music_neutral = MusicStyle("underscore", "calm", "low", "slow", "piano")
    gen = RecordingGenerator(SR)
    c.music.generator, c.music.use_generator = gen, True
    assert c.start_neutral_music()
    c.music.tick()
    assert c.music.active["music"].source == "generator"
    vid = c.music.active["music"].voice_id
    mixer.render(256)
    voice = next(v for v in mixer.voices if v.id == vid)
    c.ai_music_style(MusicStyle("drama", "tense", "low", "slow", "cello"), scene_id=c.state.scene_id,
                     connection_id=c.state.connection_id, t_input_ref=clock.now())
    assert gen.updates == [False]  # внутри сцены — плавный переход

    clock.advance(c.scene_cut_cooldown + 1)
    assert c.scene_cut("video").result == "started"
    assert gen.updates == [False, True]  # нейтральная — сразу, без плавного перехода
    mixer.render(256)
    assert voice.env_target < 0.2 and not voice.stopping  # провал, но не тишина

    clock.advance(1)
    d = c.ai_music_style(MusicStyle("comedy", "playful", "low", "medium", "ukulele"), scene_id=c.state.scene_id,
                         connection_id=c.state.connection_id, t_input_ref=clock.now())
    assert d.result == "started" and gen.updates == [False, True, True]  # выбор в провале — тоже сразу

    clock.advance(c.music.cut_dip_hold)
    c.music.tick()
    mixer.render(256)
    assert voice.env_target == 1.0  # новая музыка поднимается
    clock.advance(c.music_first_hold + 1)
    d = c.ai_music_style(MusicStyle("romance", "sad", "low", "slow", "harp"), scene_id=c.state.scene_id,
                         connection_id=c.state.connection_id, t_input_ref=clock.now())
    assert d.result == "started" and gen.updates[-1] is False  # внутри сцены — снова плавно


def test_generator_gap_is_bridged_not_fallen_back(system, clock):
    from soundmaster.registry import MusicStyle

    make, engine, mixer, journal = system
    c = make()
    c.music_mode = "AUTO"
    gen = RecordingGenerator(SR)
    c.music.generator, c.music.use_generator = gen, True
    c.music.set_style(MusicStyle("drama", "tense", "low", "slow", "cello"), owner="ai")
    c.music.tick()
    for _ in range(3 * SR // 256):  # 2 с звука кончились — генератор «переподключается»
        mixer.render(256)
    c.music.tick()
    assert c.music.active["music"].source == "generator"
    assert any(r.get("action") == "generator_bridge" for r in journal.records)
    assert not any(r.get("action") == "fallback" for r in journal.records)


def test_pool_spec_is_solo_and_varied(registry):
    assert len(registry.pool) >= len(registry.vocabulary.genres) - 1  # без пула — резерв из нейтрального
    for genre, tracks in registry.pool.items():
        assert len(tracks) >= 4
        assert all(t.style.instrument != "auto" and t.style.genre == genre for t in tracks)
    # 70% сцен — обычные современные: нейтральный фон больше остальных и на разных инструментах
    neutral = registry.pool["underscore"]
    assert len(neutral) >= 12 and len({t.style.instrument for t in neutral}) == len(neutral)
    varied = [g for g, ts in registry.pool.items() if len({t.style.instrument for t in ts}) >= 3]
    assert len(varied) >= len(registry.pool) - 3  # почти везде в жанре разные инструменты


def test_make_loop_is_seamless_and_pool_is_written(registry, tmp_path):
    import shutil
    from dataclasses import replace

    from soundmaster.music import pool as mp

    sr = mp.SR
    t = np.arange(int(10 * sr)) / sr
    x = np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32) * 0.3
    loop = mp.make_loop(x, 5.0, sr, xfade=1.0)
    assert len(loop) == 5 * sr
    assert abs(float(loop[0, 0]) - float(x[5 * sr, 0])) < 0.05  # начало петли продолжает её конец
    if shutil.which("ffmpeg") is None:
        pytest.skip("нет ffmpeg")
    reg = replace(registry, sounds_dir=tmp_path)
    calls = []

    async def fake(preset, seconds, key):
        calls.append(preset)
        return np.tile(x, (int(seconds // 10) + 2, 1))

    rep = mp.make_pool(reg, seconds=3, parallel=2, only="folk", progress=lambda s: None, recorder=fake)
    assert len(rep.made) == len(reg.pool["folk"]) and not rep.errors
    assert (tmp_path / "pool/folk/2-balalaika.flac").exists()
    assert any("balalaika" in p["text"] for p in calls[0]["prompts"] + [p for c in calls for p in c["prompts"]])
    again = mp.make_pool(reg, seconds=3, only="folk/2", progress=lambda s: None, recorder=fake)
    assert again.skipped == ["pool/folk/2-balalaika.flac"] and not again.made


def test_pool_elevenlabs_plan_and_stop_on_fatal(registry, tmp_path):
    import shutil
    from dataclasses import replace

    from soundmaster.music import pool as mp

    if shutil.which("ffmpeg") is None:
        pytest.skip("нет ffmpeg")
    noir = registry.pool["noir"][0]
    c = mp.eleven_plan(registry, noir, 34.5)["chunks"][0]
    assert c["duration_ms"] == 34500 and "66 BPM" in c["positive_styles"]
    assert any("saxophone" in x for x in c["positive_styles"])
    assert not any(x.lower().startswith("no ") for x in c["positive_styles"])  # запреты — только отдельно
    assert {"drums", "vocals", "chords", "crescendo"} <= set(c["negative_styles"])
    club = mp.eleven_plan(registry, registry.pool["electronic"][0], 10)["chunks"][0]
    assert "drums" not in club["negative_styles"] and "hi-hats" in club["negative_styles"]
    assert "100 BPM" in " ".join(club["positive_styles"]) or "110 BPM" in " ".join(club["positive_styles"])

    reg = replace(registry, sounds_dir=tmp_path)
    sr = mp.SR
    x = (np.random.default_rng(1).standard_normal((int(40 * sr), 2)) * 0.05).astype(np.float32)
    plans = []

    async def fake(plan, key):
        plans.append(plan)
        if len(plans) > 2:
            raise mp.ElevenFatal("ElevenLabs 401: quota_exceeded")
        return x

    rep = mp.make_pool(reg, seconds=3, parallel=1, only="underscore", progress=lambda s: None,
                       recorder=fake, takes=1, engine="elevenlabs")
    assert len(rep.made) == 2 and len(plans) == 3  # после «кредиты кончились» больше не просим
    assert len(rep.errors) == len(reg.pool["underscore"]) - 2
    meta = (tmp_path / reg.pool["underscore"][0].file).with_suffix(".json").read_text(encoding="utf-8")
    assert "music_v2_5" in meta and "composition_plan" in meta


def _fake_pool(registry, genre, sr=SR):
    import soundfile

    for t in registry.pool[genre]:
        path = registry.sounds_dir / t.file
        path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(path), np.full((sr, 2), 0.05 * t.n, np.float32), sr, format="FLAC")


def _wait(music, pred, timeout=5.0):
    t0 = time.time()
    while not pred():
        music.tick()
        assert time.time() - t0 < timeout, "мелодия пула не загрузилась"
        time.sleep(0.01)


def test_pool_button_plays_next_track_with_crossfade(system, registry):
    make, engine, mixer, journal = system
    c = make()
    _fake_pool(registry, "folk")
    assert c.manual_pool("folk") == "pool_loading"
    _wait(c.music, lambda: "music" in c.music.active)
    cur = c.music.active["music"]
    assert cur.state.id == "pool:folk/1" and cur.source == "file" and cur.owner == "manual"
    assert "балалайка" not in cur.state.label and "1/4" in cur.state.label
    first_voice = cur.voice_id
    for _ in range(SR // 256):
        mixer.render(256)  # первая успела зазвучать
    c.manual_pool("folk")  # повторно — следующая
    _wait(c.music, lambda: c.music.active["music"].state.id == "pool:folk/2")
    mixer.render(64)
    old = next(v for v in mixer.voices if v.id == first_voice)
    assert old.stopping and old.env_step < 1.0 / 64  # уходит плавно, а не обрывается
    for _ in range(len(registry.pool["folk"]) - 1):
        c.manual_pool("folk")
        c.music.tick()
    _wait(c.music, lambda: c.music.active["music"].state.id == "pool:folk/1")  # по кругу


def test_pool_is_reserve_when_generator_is_down(system, registry, clock):
    from soundmaster.registry import MusicStyle

    make, engine, mixer, journal = system
    c = make()
    c.music_mode = "AUTO"
    _fake_pool(registry, "folk")
    c.music.generator = BrokenGenerator(SR)
    c.music.set_generator_enabled(True)
    c.music.generator.error = "нет интернета"
    style = MusicStyle("folk", "happy", "medium", "medium", "balalaika")
    d = c.ai_music_style(style, scene_id=c.state.scene_id, connection_id=c.state.connection_id,
                         t_input_ref=clock.now())
    assert d.result == "started"
    _wait(c.music, lambda: c.music.channel_source("music") == "file")
    cur = c.music.active["music"]
    assert cur.state.id == f"style:{style.id}"  # для модели это её выбор — не «перевыбирает»
    assert cur.state.file == "pool/folk/2-balalaika.flac"  # тот же инструмент из пула
    assert any(r.get("action") == "pool_fallback" for r in journal.records)


def test_manual_pool_holds_until_scene_cut_then_automation_takes_over(system, registry, clock):
    from soundmaster.registry import MusicStyle

    make, engine, mixer, journal = system
    c = make()
    c.music_mode, c.scene_cut_mode, c.scene_cut_music = "AUTO", "AUTO", "neutral"
    c.music_neutral = MusicStyle("underscore", "calm", "low", "slow", "piano")
    _fake_pool(registry, "noir")
    c.manual_pool("noir")
    _wait(c.music, lambda: "music" in c.music.active)
    d = c.ai_music_style(MusicStyle("comedy", "playful", "low", "medium", "ukulele"), scene_id=c.state.scene_id,
                         connection_id=c.state.connection_id, t_input_ref=clock.now())
    assert (d.result, d.blocked_by) == ("blocked", "manual_music")
    clock.advance(c.scene_cut_cooldown + 1)
    c.scene_cut("video")
    assert c.music.active["music"].owner == "system"  # новая сцена — снова автоматика


def test_one_tempo_for_the_whole_show_and_no_drums(registry):
    from soundmaster.registry import MusicStyle

    v = registry.vocabulary
    a = v.preset(MusicStyle("noir", "tense", "low", "slow", "saxophone"))["config"]
    b = v.preset(MusicStyle("action", "tense", "high", "fast", "violin"))["config"]
    assert a["bpm"] == b["bpm"] <= 80  # bpm не меняется — Lyria не сбрасывает контекст (без обрывов)
    assert a["mute_drums"] and a["mute_bass"]
    p = v.pool_preset(MusicStyle("noir", "tense", "low", "slow", "saxophone"))
    assert p["config"]["guidance"] > 4 and any("dialogue" in x["text"] for x in p["prompts"])
    assert "guidance" not in a  # строгость — только для записи пула, вживую переходы плавные


def test_busyness_tells_calm_from_busy():
    from soundmaster.music.pool import SR, busyness

    def notes(per_sec):
        x = np.zeros(10 * SR, np.float32)
        for k in range(int(10 * per_sec)):
            i = int(k * SR / per_sec)
            x[i:i + 2000] += np.sin(2 * np.pi * 440 * np.arange(2000) / SR) * np.linspace(1, 0, 2000)
        return np.stack([x, x], axis=1)

    assert busyness(notes(1.5)) < busyness(notes(6)) - 2


def test_select_several(registry):
    from soundmaster.music.pool import select

    got = select(registry, "noir/2, folk")
    assert [f"{t.style.genre}/{t.n}" for t in got] == ["noir/2"] + [f"folk/{n}" for n in range(1, 5)]


def test_pool_spec_extra_and_genre_beat(tmp_path):
    from soundmaster.registry import load_registry

    p = tmp_path / "r.yaml"
    p.write_text(
        "music_vocabulary:\n  solo: one solo, no percussion\n  base_config: {bpm: 66, mute_drums: true}\n"
        "  pool_config: {bpm: 60}\n"
        "  genres: {noir: {}, club: {solo: synth with soft kick, config: {mute_drums: false, bpm: 100}}}\n"
        "  moods: {sad: {}}\n  energy: {low: {}}\n  tempo: {slow: {}}\n  instruments: {cello: {text: solo cello}}\n"
        "music_pool:\n  noir:\n    - sad/low/slow/cello | lonely, long notes\n",
        encoding="utf-8")
    reg = load_registry(p)
    t = reg.pool["noir"][0]
    pr = reg.vocabulary.pool_preset(t.style, t.extra)
    assert t.extra == "lonely, long notes" and {"text": "lonely, long notes", "weight": 1.2} in pr["prompts"]
    assert pr["config"]["bpm"] == 60 and pr["config"]["mute_drums"]
    from soundmaster.registry import MusicStyle

    club = reg.vocabulary.pool_preset(MusicStyle("club", "sad", "low", "slow", "cello"))
    assert club["config"]["bpm"] == 100 and club["config"]["mute_drums"] is False  # жанр сильнее общих правил
    assert any(x["text"] == "synth with soft kick" for x in club["prompts"])
    assert not any("no percussion" in x["text"] for x in club["prompts"])
