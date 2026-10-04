import json
import time
from pathlib import Path

import yaml

from soundmaster.app import App
from soundmaster.panel.server import handle_command
from soundmaster.testsounds import make_test_sounds

ROOT = Path(__file__).resolve().parent.parent


def make_app(tmp_path) -> App:
    cfg = yaml.safe_load((ROOT / "config" / "show.yaml").read_text(encoding="utf-8"))
    reg = yaml.safe_load((ROOT / "config" / "registry.yaml").read_text(encoding="utf-8"))
    reg["sounds_dir"] = str(tmp_path / "sounds")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "recorder.yaml").write_text(
        (ROOT / "config" / "inputs" / "recorder.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "registry.yaml").write_text(yaml.safe_dump(reg, allow_unicode=True), encoding="utf-8")
    cfg["recording"]["dir"] = str(tmp_path / "rec")
    cfg["music"]["generator"] = "none"  # тесты без сети: Lyria не подключаем
    (tmp_path / "show.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    app = App(tmp_path / "show.yaml", null_audio=True)
    make_test_sounds(app.registry)
    return app


def test_panel_commands_end_to_end(tmp_path):
    app = make_app(tmp_path)
    app.start()
    try:
        assert handle_command(app, {"cmd": "play", "effect": "doorbell"})["result"] == "played"
        assert "error" in handle_command(app, {"cmd": "bogus"})
        assert handle_command(app, {"cmd": "music", "state": "rain"})["result"] == "file"
        handle_command(app, {"cmd": "mark", "label": "звонок"})
        handle_command(app, {"cmd": "new_scene", "hint": "ресторан"})
        time.sleep(0.2)
        app.controller.tick()
        st = app.status()
        assert st["scene_id"] == 2 and st["scene_hint"] == "ресторан"
        # Новая сцена: атмосфера гаснет, музыка — нейтральная переходная (её и выберет модель потом).
        assert set(st["music"]["active"]) == {"music"}
        assert st["music"]["active"]["music"]["state"] == "style:" + app.controller.music_neutral.id
        json.dumps(st)  # состояние сериализуется для пульта
        lib = app.describe_library()
        assert all(e["available"] for e in lib["effects"])
    finally:
        app.stop()
    journal = [json.loads(line) for line in (app.run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {r["kind"] for r in journal}
    assert {"start", "decision", "audio_start", "mark", "track_start", "stop"} <= kinds
    assert (app.run_dir / "output_reference.wav").stat().st_size > 1000


def test_input_failure_keeps_panel_working(tmp_path, monkeypatch):
    from soundmaster.audio import devices

    def fail(self):
        raise RuntimeError("устройство не найдено")

    monkeypatch.setattr(devices.SoundDeviceInput, "start", fail)
    app = make_app(tmp_path)
    app.input_enabled = True
    app.start()
    try:
        assert app.status()["input"]["error"]
        assert handle_command(app, {"cmd": "play", "effect": "knock"})["result"] == "played"
    finally:
        app.stop()


def test_input_override_by_name():
    from soundmaster.app import load_show_config, resolve_input_profile
    from soundmaster.audio.inputs import load_input_profile

    cfg = load_show_config(ROOT / "config" / "show.yaml")
    assert resolve_input_profile(cfg).name == "recorder.yaml"
    mac = load_input_profile(resolve_input_profile(cfg, "mac-mic"))
    assert mac.device_channels == 1 and not mac.clean_actor_feed
    assert load_input_profile(resolve_input_profile(cfg, "brio-mic")).device_channels == 2
    import pytest

    with pytest.raises(FileNotFoundError):
        resolve_input_profile(cfg, "nope")


def test_live_listening_path(tmp_path):
    """Живой режим: звук из потока входа → фразы → распознавание → правила и модель → контроллер."""
    import asyncio
    import threading

    import numpy as np

    from soundmaster.brain.providers import FakeBrain
    from soundmaster.brain.scheduler import BrainScheduler, SchedulerConfig
    from soundmaster.fastrules import FastRules
    from soundmaster.listener import Listener
    from soundmaster.media.asr import ASRWorker, ScriptedASR

    app = make_app(tmp_path)
    app.start()
    t0 = app.clock.now()
    asr = ScriptedASR([(t0 + 0.2, t0 + 1.2, "Сейчас вылетит птичка"), (t0 + 1.8, t0 + 2.6, "Кто-то звонит в дверь")])
    brain = FakeBrain(app.registry, rules=[("звонит в дверь", {"type": "effect", "id": "doorbell",
                                                               "reason": "r", "confidence": "high"})])
    app.scheduler = BrainScheduler(brain, app.controller, app.journal, app.clock, SchedulerConfig(min_interval=0))
    app.listener = Listener(app.controller, app.journal, app.clock, None,
                            asr_worker=ASRWorker(asr, app.clock, lambda tr: None),
                            rules=FastRules(app.registry), scheduler=app.scheduler)
    sr = 48000

    def speak():
        import time

        def tone(sec):
            t = np.arange(int(sec * sr)) / sr
            return (0.25 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        sig = np.concatenate([np.zeros(int(0.2 * sr), np.float32), tone(1.0), np.zeros(int(0.6 * sr), np.float32),
                              tone(0.8), np.zeros(int(1.0 * sr), np.float32)])
        blk = sr // 10
        for i in range(0, len(sig), blk):
            target = t0 + i / sr
            time.sleep(max(0.0, target - app.clock.now()))
            app.listener.feed_audio(sig[i:i + blk], t0 + i / sr, sr)

    async def main():
        tasks = app.attach_loop(asyncio.get_running_loop())
        th = threading.Thread(target=speak)
        th.start()
        for _ in range(80):
            app.controller.tick()
            await asyncio.sleep(0.05)
        th.join()
        for t in tasks:
            t.cancel()
        await app.detach_loop()

    try:
        asyncio.run(main())
        st = app.status()
        assert [r["text"] for r in st["listen"]["recent"]] == ["Сейчас вылетит птичка", "Кто-то звонит в дверь"]
        calls = [(r.get("call"), r.get("effect"), r.get("result")) for r in app.journal.records if r["kind"] == "decision"]
        assert ("arm", "camera_shutter", "armed") in calls
        assert ("suggest_sfx", "doorbell", "suggested") in calls
    finally:
        app.stop()
