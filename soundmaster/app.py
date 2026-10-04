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
from .brain.providers import make_brain
from .brain.scheduler import BrainScheduler, SchedulerConfig
from .clock import Clock
from .controller import Controller
from .engine import Engine, LoadReport
from .fastrules import FastRules
from .listener import ASR_RATE, Listener
from .media.asr import ASRWorker, make_asr
from .media.ffmpeg import CameraReader
from .media.scenecut import SceneCutConfig, SceneCutDetector
from .media.vad import EnergyVAD
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


def resolve_input_profile(cfg: ShowConfig, override: str | None = None) -> Path:
    """Профиль из show.yaml или из --input: имя (mac-mic) или путь к YAML."""
    if not override:
        return cfg.rel(cfg.raw["input_profile"])
    p = Path(override)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    candidate = cfg.rel(f"inputs/{override}.yaml")
    if not candidate.exists():
        names = sorted(x.stem for x in cfg.rel("inputs").glob("*.yaml"))
        raise FileNotFoundError(f"профиль входа {override!r} не найден. Есть: {', '.join(names)}")
    return candidate


def load_show_config(path: str | Path) -> ShowConfig:
    path = Path(path).resolve()
    return ShowConfig(path, yaml.safe_load(path.read_text(encoding="utf-8")) or {})


class InputMonitor:
    """Забирает блоки входа из очереди захвата: запись + уровни по ролям для пульта."""

    def __init__(self, capture: SoundDeviceInput, profile: InputProfile, recorder: Recorder | None,
                 listener=None):
        self.capture = capture
        self.listener = listener
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
            if self.listener is not None and self.listener.loop is not None:
                try:
                    self.listener.feed_audio(self.profile.ai_feed(block), t, self.profile.sample_rate)
                except Exception as e:  # noqa: BLE001 — слух не должен мешать записи и звуку
                    self.listener.journal.write("error", where="listener_audio", error=f"{type(e).__name__}: {e}")
            levels = {}
            for ch in self.profile.channels:
                rms = float(np.sqrt(np.mean(block[:, ch.index] ** 2)) + 1e-12)
                levels[f"{ch.index + 1}:{ch.name or ch.role}"] = round(20 * np.log10(rms), 1)
            self.levels = levels

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


class App:
    def __init__(self, config_path: str | Path, *, null_audio: bool = False, no_input: bool = False,
                 input_profile: str | None = None):
        self.cfg = load_show_config(config_path)
        self.clock = Clock()
        self.show_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.registry: Registry = load_registry(self.cfg.rel(self.cfg.raw.get("registry", "registry.yaml")))
        self.profile: InputProfile = load_input_profile(resolve_input_profile(self.cfg, input_profile))
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

        sc = self.cfg.raw.get("scene_cut") or {}
        self.controller = Controller(self.registry, self.engine, self.music, self.journal, self.clock,
                                     clean_actor_feed=self.profile.clean_actor_feed,
                                     scene_cut_mode=str(sc.get("mode", "SUGGEST")),
                                     scene_cut_music=str(sc.get("music", "stop")),
                                     scene_cut_cooldown=float(sc.get("cooldown", 10)))
        self._build_listener(null_audio)

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

    def _build_listener(self, null_audio: bool) -> None:
        """Слух (распознавание + правила), «мозг» и камера — если включены в show.yaml."""
        listen = self.cfg.raw.get("listen") or {}
        bcfg = self.cfg.raw.get("brain") or {}
        vcfg = self.cfg.raw.get("video") or {}
        sc = self.cfg.raw.get("scene_cut") or {}
        self.listener: Listener | None = None
        self.scheduler: BrainScheduler | None = None
        self.camera: CameraReader | None = None
        self.listen_error: str | None = None
        asr = None
        try:
            asr = make_asr(listen.get("asr") or "off", listen)
        except Exception as e:  # noqa: BLE001 — без распознавания пульт всё равно работает
            self.listen_error = f"распознавание: {e}"
        brain = make_brain(bcfg.get("provider") or "off", self.registry, bcfg)
        video_on = bool(vcfg.get("enabled", False)) and not null_audio
        if not (asr or brain or video_on):
            return
        if brain:
            self.scheduler = BrainScheduler(brain, self.controller, self.journal, self.clock,
                                            SchedulerConfig.from_dict(bcfg))
        scenecut = (SceneCutDetector(SceneCutConfig.from_dict(sc.get("detector")))
                    if video_on and str(sc.get("mode", "SUGGEST")).upper() != "OFF" else None)
        self.listener = Listener(
            self.controller, self.journal, self.clock, None,
            vad=EnergyVAD(ASR_RATE, **(listen.get("vad") or {})),
            asr_worker=ASRWorker(asr, self.clock, lambda tr: None) if asr else None,
            rules=FastRules(self.registry), scheduler=self.scheduler, scenecut=scenecut,
        )
        if video_on:
            self.camera = CameraReader(str(vcfg.get("camera", "0")), float(vcfg.get("fps", 5)),
                                       int(vcfg.get("width", 640)), int(vcfg.get("height", 360)),
                                       self.listener.feed_frame, self.clock,
                                       capture_size=str(vcfg.get("capture_size", "1280x720")))

    def attach_loop(self, loop) -> list:
        """Запуск слуха внутри asyncio-цикла пульта. Возвращает фоновые задачи."""
        tasks = []
        if self.listener:
            self.listener.loop = loop
            if self.listener.asr_worker:
                self.listener.asr_worker.start()  # модель загружается в потоке распознавания
            if self.camera:
                try:
                    self.camera.start()
                except Exception as e:  # noqa: BLE001
                    self.camera.error = f"{type(e).__name__}: {e}"
        if self.scheduler:
            tasks.append(loop.create_task(self.scheduler.run()))
        self.journal.write("listen", asr=self.listener.asr_worker.asr.name
                           if self.listener and self.listener.asr_worker else None,
                           brain=f"{self.scheduler.brain.name}:{self.scheduler.brain.model}" if self.scheduler else None,
                           camera=bool(self.camera), error=self.listen_error)
        return tasks

    async def detach_loop(self) -> None:
        if self.camera:
            self.camera.stop()
        if self.listener and self.listener.asr_worker:
            self.listener.asr_worker.stop()
        if self.scheduler:
            await self.scheduler.brain.close()

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
                self.monitor = InputMonitor(self.capture, self.profile, self.recorder, self.listener)
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
        s["listen"] = {
            "enabled": self.listener is not None,
            "asr": self.listener.asr_worker.asr.name if self.listener and self.listener.asr_worker else None,
            "recent": self.listener.recent if self.listener else [],
            "speech": bool(self.listener and self.listener.vad.in_speech),
            "asr_state": self.listener.asr_worker.state if self.listener and self.listener.asr_worker else None,
            "error": self.listen_error or (self.listener.asr_worker.last_error
                                           if self.listener and self.listener.asr_worker else None),
            "camera_error": self.camera.error if self.camera else None,
            "brain": self.scheduler.status() if self.scheduler else None,
        }
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
