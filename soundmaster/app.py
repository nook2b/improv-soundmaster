"""Сборка системы из config/show.yaml."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import __version__
from .audio.devices import NullOutput, SoundDeviceInput, SoundDeviceOutput
from .audio.inputs import InputProfile, load_input_profile
from .audio.mixer import Mixer
from .audio.recording import Recorder
from .clock import Clock
from .controller import Controller
from .engine import Engine, LoadReport
from .journal import Journal
from .music.director import MusicDirector
from .music.generators import make_generator
from .registry import Registry, load_registry


@dataclass
class ShowConfig:
    path: Path
    raw: dict[str, Any]

    def rel(self, p: str) -> Path:
        return (self.path.parent / p).resolve()

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return (self.raw.get(section) or {}).get(key, default)


def load_show_config(path: str | Path) -> ShowConfig:
    path = Path(path).resolve()
    return ShowConfig(path, yaml.safe_load(path.read_text(encoding="utf-8")) or {})


class InputMonitor:
    """Забирает блоки входа из очереди захвата: запись + уровни по ролям для пульта."""

    def __init__(self, capture: SoundDeviceInput, profile: InputProfile, recorder: Recorder | None):
        self.capture = capture
        self.profile = profile
        self.track = recorder.track("input", profile.device_channels, profile.sample_rate) if recorder else None
        self.levels: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="input-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                block, t = self.capture.blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.track:
                self.track.push(block, t)
            levels = {}
            for ch in self.profile.channels:
                rms = float(np.sqrt(np.mean(block[:, ch.index] ** 2)) + 1e-12)
                levels[f"{ch.index + 1}:{ch.name or ch.role}"] = round(20 * np.log10(rms), 1)
            self.levels = levels

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


class App:
    def __init__(self, config_path: str | Path, *, null_audio: bool = False, no_input: bool = False):
        self.cfg = load_show_config(config_path)
        self.clock = Clock()
        self.show_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.registry: Registry = load_registry(self.cfg.rel(self.cfg.raw.get("registry", "registry.yaml")))
        self.profile: InputProfile = load_input_profile(self.cfg.rel(self.cfg.raw["input_profile"]))
        self.input_enabled = bool(self.cfg.raw.get("input_enabled", True)) and not no_input and not null_audio

        rec_enabled = bool(self.cfg.get("recording", "enabled", True))
        self.run_dir = self.cfg.rel(self.cfg.get("recording", "dir", "../recordings")) / self.show_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.run_dir / "journal.jsonl", self.clock,
                               base={"show_id": self.show_id})
        self.recorder = Recorder(self.run_dir, self.journal) if rec_enabled else None

        sr = int(self.cfg.get("output", "sample_rate", 48000))
        self.mixer = Mixer(sample_rate=sr)
        self.mixer.set_master_gain_db(float(self.cfg.get("output", "master_gain_db", 0)))
        self.engine = Engine(self.registry, self.mixer, self.clock, self.journal)

        generator = make_generator(self.cfg.get("music", "generator"), sr)
        self.music = MusicDirector(self.registry, self.engine, self.journal, generator)
        self.music.set_generator_enabled(bool(self.cfg.get("music", "generator_enabled", False)))

        self.controller = Controller(self.registry, self.engine, self.music, self.journal, self.clock,
                                     clean_actor_feed=self.profile.clean_actor_feed)

        ref_track = self.recorder.track("output_reference", 2, sr) if self.recorder else None

        def on_block(block: np.ndarray, t_out: float) -> None:
            self.engine.on_output_block(block, t_out)
            if ref_track:
                ref_track.push(block, t_out)

        blocksize = int(self.cfg.get("output", "blocksize", 256))
        if null_audio:
            self.output = NullOutput(self.mixer, self.clock, blocksize, on_block)
        else:
            self.output = SoundDeviceOutput(self.mixer, self.clock, self.cfg.get("output", "device"),
                                            blocksize, on_block=on_block)
        self.capture: SoundDeviceInput | None = None
        self.monitor: InputMonitor | None = None
        self.load_report: LoadReport | None = None
        self.input_error: str | None = None

    def start(self) -> None:
        self.journal.write("start", version=__version__, input_profile=self.profile.name,
                           clean_actor_feed=self.profile.clean_actor_feed,
                           wall_start=self.clock.wall_start, config=str(self.cfg.path.name))
        self.load_report = self.engine.preload()
        self.output.start()
        self.journal.write("output", latency=getattr(self.output, "output_latency", None),
                           sample_rate=self.mixer.sample_rate)
        if self.input_enabled:
            # Ошибка входа не должна ронять ручной пульт: звук в зал важнее записи.
            try:
                self.capture = SoundDeviceInput(self.profile, self.clock)
                self.capture.start()
                self.monitor = InputMonitor(self.capture, self.profile, self.recorder)
                self.monitor.start()
            except Exception as e:  # noqa: BLE001
                self.capture = None
                self.input_error = f"{type(e).__name__}: {e}"
                self.journal.write("input_error", error=self.input_error)

    def stop(self) -> None:
        self.music.stop_all()
        time.sleep(0.05)
        if self.capture:
            self.capture.stop()
        if self.monitor:
            self.monitor.stop()
        self.output.stop()
        self.controller.tick()
        if self.recorder:
            self.recorder.close()
        self.journal.write("stop")
        self.journal.close()

    def status(self) -> dict[str, Any]:
        s = self.controller.snapshot()
        s["input"] = {
            "profile": self.profile.name,
            "enabled": self.input_enabled,
            "levels": self.monitor.levels if self.monitor else {},
            "overflows": self.capture.overflows if self.capture else 0,
            "error": self.input_error,
        }
        s["output"] = {"xruns": self.output.xruns, "limited_blocks": self.mixer.limited_blocks,
                       "latency_ms": round(1000 * float(getattr(self.output, "output_latency", 0.0)), 1)}
        s["recording"] = str(self.run_dir) if self.recorder else None
        return s

    def describe_library(self) -> dict[str, Any]:
        return {
            "effects": [
                {"id": e.id, "label": e.label, "hotkey": e.hotkey, "channel": e.channel, "class": e.cls,
                 "risk": e.risk, "available": self.engine.available(e)}
                for e in self.registry.effects.values()
            ],
            "music": [
                {"id": m.id, "label": m.label, "hotkey": m.hotkey, "channel": m.channel,
                 "has_file": bool(m.file and m.file in self.engine.sounds), "has_generator": bool(m.generator)}
                for m in self.registry.music_states.values()
            ],
            "missing": self.load_report.missing if self.load_report else [],
            "errors": self.load_report.errors if self.load_report else [],
        }
