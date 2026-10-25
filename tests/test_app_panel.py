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


def test_run_flags_override_show_yaml():
    from soundmaster.app import apply_run_overrides, load_show_config

    cfg = load_show_config(ROOT / "config" / "show.yaml")
    apply_run_overrides(cfg, asr="faster", asr_model="large-v3-turbo", brain="openai", camera="Logitech BRIO")
    assert cfg.raw["listen"]["asr"] == "faster" and cfg.raw["listen"]["faster_model"] == "large-v3-turbo"
    assert cfg.raw["brain"]["provider"] == "openai"
    assert cfg.raw["video"] == {**cfg.raw["video"], "enabled": True, "camera": "Logitech BRIO"}


def test_panel_genre_button_plays_pool(tmp_path):
    import numpy as np
    import soundfile

    app = make_app(tmp_path)
    for t in app.registry.pool["noir"]:
        p = app.registry.sounds_dir / t.file
        p.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(p), np.full((48000, 2), 0.05, np.float32), 48000, format="FLAC")
    app.start()
    try:
        lib = app.describe_library()
        noir = next(g for g in lib["pool"] if g["genre"] == "noir")
        assert noir["available"] == noir["total"] == len(app.registry.pool["noir"])
        assert next(g for g in lib["pool"] if g["genre"] == "folk")["available"] == 0
        assert handle_command(app, {"cmd": "pool", "genre": "noir"})["result"] in ("pool", "pool_loading")
        t0 = time.time()
        while (app.status()["music"]["active"].get("music", {}).get("state") != "pool:noir/1"
               and time.time() - t0 < 5):
            app.controller.tick()
            time.sleep(0.02)
        cur = app.status()["music"]["active"]["music"]
        assert cur["state"] == "pool:noir/1" and f"1/{noir['total']}" in cur["label"]
        assert handle_command(app, {"cmd": "pool", "genre": "folk"})["result"] == "no_pool"
    finally:
        app.stop()


def test_scene_cut_detector_built_even_when_mode_off(tmp_path):
    """«Монтаж» можно включить с пульта: детектор есть при любой камере, режим проверяет контроллер."""
    app = make_app(tmp_path)
    app.cfg.raw["video"] = {"enabled": True, "camera": "0"}
    app.cfg.raw["scene_cut"] = {**(app.cfg.raw.get("scene_cut") or {}), "mode": "OFF"}
    app._build_listener(null_audio=False)
    assert app.listener is not None and app.listener.scenecut is not None
    assert "error" in _safe(app, {"cmd": "scene_cut_mode", "mode": "ON"})
    assert handle_command(app, {"cmd": "scene_cut_mode", "mode": "auto"})["result"] == "AUTO"
    app.journal.close()


def _safe(app, msg):
    try:
        return handle_command(app, msg) or {}
    except ValueError as e:
        return {"error": str(e)}


def test_panel_ticker_survives_dead_and_slow_clients(tmp_path, monkeypatch):
    """Оборванный или зависший клиент пульта не останавливает tick (подсказки истекают, музыка грузится)."""
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from soundmaster.panel import server

    app = make_app(tmp_path)
    app.start()
    ticks = []
    orig_tick = app.controller.tick
    monkeypatch.setattr(app.controller, "tick", lambda: (ticks.append(1), orig_tick()))
    monkeypatch.setattr(server, "SEND_TIMEOUT", 0.2)
    mode = {"v": "ok"}
    orig_send = web.WebSocketResponse.send_str

    async def send_str(self, data, compress=None):
        if mode["v"] == "dead":
            raise ConnectionResetError("клиент пропал")
        if mode["v"] == "slow":
            await asyncio.sleep(3600)
        return await orig_send(self, data, compress)

    monkeypatch.setattr(web.WebSocketResponse, "send_str", send_str)

    async def main():
        client = TestClient(TestServer(server.make_web_app(app)))
        await client.start_server()
        try:
            for m in ("dead", "slow"):
                ws = await client.ws_connect("/ws")
                await asyncio.sleep(0.3)
                mode["v"] = m
                n0 = len(ticks)
                await asyncio.sleep(0.8)
                assert len(ticks) - n0 >= 8, m  # цикл пульта идёт дальше
                mode["v"] = "ok"
                await ws.close()
            assert (await client.get("/library")).status == 200
        finally:
            await client.close()

    try:
        asyncio.run(main())
    finally:
        app.stop()
