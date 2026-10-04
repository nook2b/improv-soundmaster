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
        assert st["music"]["active"] == {}
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
