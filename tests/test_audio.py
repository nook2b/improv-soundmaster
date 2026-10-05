from pathlib import Path

import numpy as np
import pytest

from soundmaster.audio.inputs import InputProfileError, load_input_profile
from soundmaster.audio.mixer import Mixer, StreamBuffer
from soundmaster.audio.wavio import load_sound, write_wav

ROOT = Path(__file__).resolve().parent.parent
SR = 8000


def tone(sec=0.5, amp=0.5):
    return np.full((int(sec * SR), 2), amp, np.float32)


def test_mixer_plays_and_ends():
    m = Mixer(sample_rate=SR)
    m.play("x", tone(0.1), "sfx", "manual")
    out = m.render(400)
    assert out[:, 0].max() > 0
    m.render(800)
    ends = [e for e in m.events if e.kind == "ended"]
    assert ends and not m.voices


def test_limiter_keeps_ceiling():
    m = Mixer(sample_rate=SR, ceiling_db=-1.0)
    m.play("a", tone(amp=0.9), "sfx", "manual")
    m.play("b", tone(amp=0.9), "sting", "manual")
    out = m.render(256)
    assert np.abs(out).max() <= m.ceiling + 1e-6


def test_stop_by_owner_with_fade():
    m = Mixer(sample_rate=SR)
    m.play("ai", tone(2), "sfx", "ai")
    m.play("man", tone(2), "sting", "manual")
    m.render(128)
    m.stop(owner="ai", fade=0.01)
    m.render(256)
    assert {v.owner for v in m.voices} == {"manual"}


def test_bus_max_simultaneous_replaces_music():
    m = Mixer(sample_rate=SR)
    m.play("m1", tone(2), "music", "manual")
    m.render(64)
    m.play("m2", tone(2), "music", "manual")
    m.render(SR // 10)
    assert [v.effect_id for v in m.voices] == ["m2"]


def test_ducking_lowers_music_under_sfx():
    m = Mixer(sample_rate=SR)
    m.play("m", tone(3, 0.2), "music", "manual")
    level_before = np.abs(m.render(SR // 2)).mean()
    m.play("s", np.zeros((SR, 2), np.float32), "sfx", "manual")  # тихий эффект, но шина активна
    level_during = np.abs(m.render(SR // 2)).mean()
    assert level_during < level_before * 0.6


def test_stream_buffer_underrun_counted():
    b = StreamBuffer(1.0, SR)
    b.push(tone(0.01))
    out, _ = b.read(200)
    assert b.underruns == 1 and out[:80].mean() > 0


def test_wav_roundtrip_and_resample(tmp_path):
    p = tmp_path / "x.wav"
    write_wav(p, np.full((SR, 1), 0.25, np.float32), SR)
    data = load_sound(p, 2 * SR)
    assert data.shape == (2 * SR, 2)
    assert abs(float(data.mean()) - 0.25) < 0.01


def test_recorder_profile_has_no_clean_feed():
    p = load_input_profile(ROOT / "config" / "inputs" / "recorder.yaml")
    assert not p.clean_actor_feed and p.ai_feed_roles == ("stage_mix",)
    block = np.ones((10, 2), np.float32)
    assert p.ai_feed(block).shape == (10,)


def test_future_profiles_have_clean_feed():
    lav = load_input_profile(ROOT / "config" / "inputs" / "lavaliers-interface.yaml")
    assert lav.clean_actor_feed and lav.ai_feed_roles == ("actor_mic",) and lav.device_channels == 10
    block = np.zeros((4, 10), np.float32)
    block[:, 8] = 1.0  # зал не должен попасть во вход AI
    assert lav.ai_feed(block).max() == 0
    assert lav.room_feed(block).max() == 1.0
    aux = load_input_profile(ROOT / "config" / "inputs" / "mixer-aux.yaml")
    assert aux.ai_feed_roles == ("actors_aux",)


def test_bad_profile_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("channels:\n  - {channel: 1, role: guitar}\n", encoding="utf-8")
    with pytest.raises(InputProfileError):
        load_input_profile(p)


def test_stream_buffer_bridges_a_gap_with_recent_music_then_crossfades_back():
    from soundmaster.audio.mixer import StreamBuffer

    sr = 1000
    b = StreamBuffer(8, sr, bridge_seconds=2, bridge_max=5, bridge_xfade=0.1)
    music = np.stack([np.linspace(0.1, 0.5, 3 * sr)] * 2, axis=1).astype(np.float32)
    b.push(music)
    out, _ = b.read(3 * sr)
    assert np.allclose(out, music)
    gap, _ = b.read(sr)  # генератор молчит: играет петля из недавнего, не тишина
    assert b.bridging and b.bridges == 1 and b.underruns == 0
    assert np.abs(gap).min() > 0.1 and np.abs(gap).max() <= 0.5 + 1e-6
    b.push(np.full((sr, 2), 0.9, np.float32))
    back, _ = b.read(sr)
    assert not b.bridging
    assert back[0, 0] < 0.9 and np.allclose(back[200:], 0.9)  # плавный уход с петли за 0,1 с
    b.read(10 * sr)  # долгий обрыв: мостик не дольше bridge_max — дальше пусто, режиссёр уйдёт на резерв
    assert b.underruns > 0
