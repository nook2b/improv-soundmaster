import numpy as np

from soundmaster.clock import FakeClock
from soundmaster.media.asr import ASRWorker, ScriptedASR, clean_text, load_transcript_file
from soundmaster.media.scenecut import SceneCutDetector
from soundmaster.media.vad import EnergyVAD

SR = 16000


def speech_like(sec, amp=0.2, f=220):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def silence(sec, amp=0.002):
    return (amp * np.random.default_rng(1).standard_normal(int(sec * SR))).astype(np.float32)


def run_vad(signal, vad, block=1600):
    out = []
    for i in range(0, len(signal), block):
        out += vad.push(signal[i:i + block], i / SR)
    out += vad.flush(len(signal) / SR)
    return out


def test_vad_finds_two_phrases():
    sig = np.concatenate([silence(1.0), speech_like(1.0), silence(0.8), speech_like(1.5), silence(1.0)])
    utts = run_vad(sig, EnergyVAD(SR))
    assert len(utts) == 2
    assert abs(utts[0].t_start - 1.0) < 0.3 and abs(utts[0].t_end - 2.0) < 0.6
    assert abs(utts[1].t_start - 2.8) < 0.3


def test_vad_force_cuts_long_speech():
    sig = np.concatenate([silence(1.0), speech_like(14.0), silence(1.0)])
    utts = run_vad(sig, EnergyVAD(SR, max_utterance_s=6.0))
    assert len(utts) >= 3 and utts[0].forced_cut
    assert all(u.t_end - u.t_start <= 6.1 for u in utts)


def test_hallucinations_removed():
    assert clean_text("Продолжение следует...") == ""
    assert clean_text("  ...  ") == ""
    assert clean_text("Кто-то звонит в дверь") == "Кто-то звонит в дверь"


def test_scripted_asr_and_worker(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("# расшифровка\n0:01-0:02.5 Сейчас вылетит птичка\n4 5 чик\n", encoding="utf-8")
    lines = load_transcript_file(p)
    assert lines[0] == (1.0, 2.5, "Сейчас вылетит птичка")
    asr = ScriptedASR(lines)
    got = []
    w = ASRWorker(asr, FakeClock(0), got.append)
    sig = np.concatenate([silence(1.0), speech_like(1.5), silence(1.5), speech_like(0.6), silence(1.0)])
    for u in run_vad(sig, EnergyVAD(SR)):
        tr = w.process(u)
        if tr:
            got.append(tr)
    assert [t.text for t in got] == ["Сейчас вылетит птичка", "чик"]


def frames_with_rect(n, positions, rect_w=12, rect_h=0.8, w=96, h=54, bg=30, fg=200, light_at=None):
    out = []
    for i in range(n):
        f = np.full((h, w, 3), bg, np.uint8)
        x = positions(i)
        if x is not None:
            x0 = int(x * (w - rect_w))
            y0 = int(h * (1 - rect_h))
            f[y0:, x0:x0 + rect_w] = fg
        if light_at is not None and i >= light_at:
            f = np.clip(f.astype(int) + 120, 0, 255).astype(np.uint8)
        out.append(f)
    return out


def detect(frames, fps=10.0):
    d = SceneCutDetector()
    events = []
    for i, f in enumerate(frames):
        e = d.push(i / fps, f)
        if e:
            events.append(e)
    # Пустые кадры в конце закрывают трек.
    for j in range(10):
        e = d.push((len(frames) + j) / fps, frames[0] * 0 + 30)
        if e:
            events.append(e)
    return events


def test_scenecut_detects_fast_sweep():
    frames = frames_with_rect(25, lambda i: min(1.0, i / 15) if i <= 15 else None)  # 1.5 с слева направо
    ev = detect(frames)
    assert len(ev) == 1 and ev[0].direction == "left_to_right"


def test_scenecut_detects_right_to_left():
    frames = frames_with_rect(25, lambda i: max(0.0, 1 - i / 15) if i <= 15 else None)
    ev = detect(frames)
    assert len(ev) == 1 and ev[0].direction == "right_to_left"


def test_scenecut_ignores_small_figure_far_away():
    frames = frames_with_rect(25, lambda i: min(1.0, i / 15) if i <= 15 else None, rect_h=0.2)
    assert detect(frames) == []


def test_scenecut_ignores_moving_in_place():
    frames = frames_with_rect(60, lambda i: 0.4 + 0.1 * np.sin(i / 2))
    assert detect(frames) == []


def test_scenecut_ignores_light_change():
    frames = frames_with_rect(30, lambda i: None, light_at=10)
    assert detect(frames) == []


def test_scenecut_ignores_slow_walk():
    frames = frames_with_rect(120, lambda i: min(1.0, i / 100))  # 10 с через сцену — это не монтаж
    assert detect(frames) == []


def test_url_detection():
    from soundmaster.media.download import is_url

    assert is_url("https://youtu.be/Zha1fA7l6gY?si=x")
    assert is_url("http://drive.google.com/file/d/1/view")
    assert not is_url("/Users/me/scene.mp4") and not is_url("scene.mp4")


def test_download_requires_yt_dlp(monkeypatch, tmp_path):
    import shutil as sh

    import pytest

    from soundmaster.media import download as dl

    monkeypatch.setattr(sh, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="yt-dlp"):
        dl.download("https://youtu.be/x", tmp_path)
