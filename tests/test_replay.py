import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from soundmaster.audio.wavio import write_wav
from soundmaster.brain.providers import FakeBrain
from soundmaster.registry import load_registry
from soundmaster.replay import Replay, ReplayOptions
from soundmaster.report import load_journal

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg")
SR = 16000
W, H, FPS = 160, 90, 10


def tone(sec, amp=0.25):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 200 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def quiet(sec):
    return (0.002 * np.random.default_rng(0).standard_normal(int(sec * SR))).astype(np.float32)


@pytest.fixture(scope="module")
def scene_video(tmp_path_factory):
    """12 с: реплики на 1–2.5 с («птичка»), 4–4.6 с («чик»), 6–7.5 с («звонит в дверь»);
    на 9–10.5 с — проход крупной фигуры перед камерой."""
    d = tmp_path_factory.mktemp("media")
    audio = np.concatenate([quiet(1), tone(1.5), quiet(1.5), tone(0.6), quiet(1.4), tone(1.5), quiet(4.5)])
    write_wav(d / "a.wav", audio, SR)
    frames = []
    for i in range(12 * FPS):
        f = np.full((H, W, 3), 40, np.uint8)
        t = i / FPS
        if 9.0 <= t <= 10.5:
            x = int((t - 9.0) / 1.5 * (W - 30))
            f[int(H * 0.1):, x:x + 30] = 220
        frames.append(f)
    raw = d / "v.rgb"
    raw.write_bytes(np.stack(frames).tobytes())
    out = d / "scene.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                    "-r", str(FPS), "-i", str(raw), "-i", str(d / "a.wav"), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    (d / "scene.txt").write_text(
        "0:01-0:02.5 Смотрите сюда, сейчас вылетит птичка\n"
        "0:04-0:04.6 Чик!\n"
        "0:06-0:07.5 Кажется, кто-то звонит в дверь\n", encoding="utf-8")
    return out


def run(scene_video, tmp_path, rules=None, **kw):
    registry = load_registry(ROOT / "config" / "registry.yaml")
    brain = FakeBrain(registry, rules=rules or [("звонит в дверь", {"type": "effect", "id": "doorbell",
                                                                    "reason": "звонок в дверь", "confidence": "high"})])
    opts = ReplayOptions(media=scene_video, config=ROOT / "config" / "show.yaml", speed=0,
                         transcript=scene_video.with_suffix(".txt"), out_dir=tmp_path, **kw)
    rp = Replay(opts, brain=brain)
    summary = rp.run()
    return summary, load_journal(rp.out_dir / "journal.jsonl"), rp


def test_replay_fast_end_to_end(scene_video, tmp_path):
    summary, j, rp = run(scene_video, tmp_path, scene_cut_mode="SUGGEST")
    texts = [r["text"] for r in j if r["kind"] == "transcript"]
    assert texts == ["Смотрите сюда, сейчас вылетит птичка", "Чик!", "Кажется, кто-то звонит в дверь"]
    dec = [(r["call"], r["effect"], r["result"]) for r in j if r["kind"] == "decision"]
    # «птичка» взводит затвор (в режиме SUGGEST — теневое взведение), «чик» срабатывает подсказкой.
    assert ("arm", "camera_shutter", "armed") in dec
    assert ("fire", "camera_shutter", "suggested") in dec
    # модель предложила звонок после реплики
    assert ("suggest_sfx", "doorbell", "suggested") in dec
    # проход перед камерой — подсказка новой сцены
    assert summary["scene_cuts"] == 1
    assert ("scene_cut", "new_scene", "suggested") in dec
    assert (rp.out_dir / "report.html").exists()
    # модель не видела будущих реплик: в запросе про «птичку» нет «двери»
    first_ctx = next(r["context"] for r in j if r["kind"] == "brain_request")
    assert "дверь" not in first_ctx


def test_replay_scene_cut_auto_starts_new_scene(scene_video, tmp_path):
    summary, j, _ = run(scene_video, tmp_path, scene_cut_mode="AUTO")
    assert summary["new_scenes"] == 1
    assert any(r["kind"] == "system" and r.get("action") == "new_scene" for r in j)


def test_replay_negation_blocks_arm(scene_video, tmp_path):
    txt = scene_video.with_suffix(".txt")
    neg = tmp_path / "neg.txt"
    neg.write_text(txt.read_text(encoding="utf-8").replace("сейчас вылетит птичка", "птичка не вылетит, хотя нет"),
                   encoding="utf-8")
    registry = load_registry(ROOT / "config" / "registry.yaml")
    opts = ReplayOptions(media=scene_video, config=ROOT / "config" / "show.yaml", speed=0, transcript=neg,
                         out_dir=tmp_path)
    rp = Replay(opts, brain=FakeBrain(registry))
    rp.run()
    j = load_journal(rp.out_dir / "journal.jsonl")
    assert not any(r["kind"] == "decision" and r["call"] == "arm" for r in j)
    assert any(r["kind"] == "rule_negated" for r in j)


def test_cloud_brain_refused_in_fast_mode(scene_video, tmp_path):
    from soundmaster.brain.providers import OpenAIBrain, Prices

    registry = load_registry(ROOT / "config" / "registry.yaml")
    opts = ReplayOptions(media=scene_video, config=ROOT / "config" / "show.yaml", speed=0, out_dir=tmp_path)
    with pytest.raises(ValueError):
        Replay(opts, brain=OpenAIBrain(registry, "gpt-5-mini", Prices()))


def test_replay_realtime_threads(scene_video, tmp_path):
    registry = load_registry(ROOT / "config" / "registry.yaml")
    brain = FakeBrain(registry, latency=0.3, rules=[("звонит в дверь", {"type": "effect", "id": "doorbell",
                                                                        "reason": "звонок", "confidence": "high"})])
    opts = ReplayOptions(media=scene_video, config=ROOT / "config" / "show.yaml", speed=1,
                         transcript=scene_video.with_suffix(".txt"), out_dir=tmp_path, max_seconds=8.5)
    rp = Replay(opts, brain=brain)
    summary = rp.run()
    j = load_journal(rp.out_dir / "journal.jsonl")
    assert summary["transcripts"] == 3
    assert summary["brain_latency_p50"] >= 0.3
    assert 0 <= summary["asr_delay_p50"] < 0.5
    assert any(r["kind"] == "decision" and r["effect"] == "doorbell" for r in j)


def test_cli_replay_by_url(scene_video, tmp_path, monkeypatch):
    """Регрессия: replay со ссылкой обращался к настройкам до их загрузки."""
    import shutil as sh

    import yaml

    from soundmaster import __main__ as cli
    from soundmaster.media import download as dl

    cfg = yaml.safe_load((ROOT / "config" / "show.yaml").read_text(encoding="utf-8"))
    cfg["registry"] = str(ROOT / "config" / "registry.yaml")
    cfg["input_profile"] = str(ROOT / "config" / "inputs" / "recorder.yaml")
    cfg["recording"]["dir"] = str(tmp_path / "rec")
    show = tmp_path / "show.yaml"
    show.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    def fake_download(url, directory, max_height=720):
        directory.mkdir(parents=True, exist_ok=True)
        return Path(sh.copy(scene_video, directory / "fakeid.mp4"))

    monkeypatch.setattr(dl, "download", fake_download)
    rc = cli.main(["--config", str(show), "replay", "https://youtu.be/Zha1fA7l6gY", "--fast",
                   "--transcript", str(scene_video.with_suffix(".txt"))])
    assert rc == 0
    assert (tmp_path / "rec" / "downloads" / "fakeid.mp4").exists()
    assert list((tmp_path / "rec").glob("replay_*_fakeid/report.html"))


def test_replay_music_auto_and_scene_cut_changes_music(scene_video, tmp_path):
    """Музыка от AI включается сама; «монтаж» в AUTO начинает новую сцену, музыка — в нейтральную."""
    music = {"type": "music", "id": None, "reason": "фотограф", "confidence": "high",
             "genre": "comedy", "mood": "playful", "energy": "medium", "tempo": "medium"}
    summary, j, rp = run(scene_video, tmp_path, rules=[("птичка", music)], scene_cut_mode="AUTO")
    dec = [(r["call"], r["effect"], r["result"]) for r in j if r["kind"] == "decision"]
    assert ("music_style", "comedy/playful/medium/medium", "started") in dec
    assert ("scene_cut", "new_scene", "started") in dec
    t_music = next(r["t"] for r in j if r["kind"] == "decision" and r["call"] == "music_style")
    t_cut = next(r["t"] for r in j if r["kind"] == "decision" and r["call"] == "scene_cut")
    assert t_music < t_cut
    neutral = rp.controller.music_neutral
    assert rp.music.current("music").id == f"style:{neutral.id}"
    assert rp.music.active["music"].owner == "system"
    assert any(r["kind"] == "music" and r.get("state") == f"style:{neutral.id}" and r["t"] >= t_cut for r in j)
