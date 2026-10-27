"""Прогон записи через систему «как вживую» — теневой режим: ничего не звучит, всё пишется в журнал.

- speed=1: реальный темп; задержки распознавания и модели настоящие (для оценки тайминга и цены).
- speed=0: быстрее реального времени, на подставных часах; только без облачной модели или с
  подставной — для проверки правил, детектора и контроллера.

Модель получает данные строго в порядке их появления: будущих реплик и кадров у неё нет.
"""

from __future__ import annotations

import asyncio
import heapq
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .app import autonomy_kwargs, load_show_config
from .audio.devices import NullOutput
from .audio.mixer import Mixer
from .brain.providers import Brain, make_brain
from .brain.scheduler import BrainScheduler, SchedulerConfig
from .clock import Clock, FakeClock
from .controller import Controller
from .engine import Engine
from .fastrules import FastRules
from .journal import Journal
from .listener import ASR_RATE, Listener
from .media.asr import ASR, ASRWorker, ScriptedASR, load_transcript_file, make_asr
from .media.ffmpeg import probe, read_audio_blocks, read_video_frames
from .media.peoplecut import make_scenecut
from .media.vad import EnergyVAD
from .music.director import MusicDirector
from .registry import load_registry
from .report import summarize, write_html

AUDIO_BLOCK = 1600  # 0,1 с при 16 кГц
ASR_WARMUP_TIMEOUT = 900.0  # загрузка модели распознавания (первый раз — со скачиванием)


@dataclass
class ReplayOptions:
    media: Path
    config: Path
    speed: float = 1.0
    asr: str | None = None  # None — из show.yaml
    asr_model: str | None = None
    brain: str | None = None
    transcript: Path | None = None
    scene_cut_mode: str | None = None
    scene_cut_kind: str | None = None  # people | pixel (по умолчанию — из show.yaml)
    clean_feed: bool = False
    video: bool = True
    out_dir: Path | None = None
    max_seconds: float | None = None


class Replay:
    def __init__(self, opts: ReplayOptions, *, brain: Brain | None = None, asr: ASR | None = None):
        self.opts = opts
        cfg = load_show_config(opts.config)
        self.cfg = cfg
        self.info = probe(opts.media)
        self.registry = load_registry(cfg.rel(cfg.raw.get("registry", "registry.yaml")))
        if opts.speed not in (0, 1):
            raise ValueError("speed: 1 — реальный темп, 0 — быстрый режим без облачной модели")
        self.fast = opts.speed == 0
        self.clock: Clock = FakeClock(0.0) if self.fast else Clock()

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = opts.out_dir or cfg.rel(cfg.get("recording", "dir", "../recordings"))
        self.out_dir = Path(base) / f"replay_{stamp}_{opts.media.stem}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.out_dir / "journal.jsonl", self.clock,
                               base={"run": "replay", "media": opts.media.name})

        self.mixer = Mixer(sample_rate=int(cfg.get("output", "sample_rate", 48000)))
        self.engine = Engine(self.registry, self.mixer, self.clock, self.journal)
        self.engine.preload()
        filled = self.engine.fill_missing_with_silence()
        if filled:
            self.journal.write("replay_note", silent_placeholders=filled)
        self.music = MusicDirector(self.registry, self.engine, self.journal, None)
        sc = cfg.raw.get("scene_cut") or {}
        self.controller = Controller(
            self.registry, self.engine, self.music, self.journal, self.clock,
            clean_actor_feed=opts.clean_feed,
            scene_cut_mode=opts.scene_cut_mode or sc.get("mode", "SUGGEST"),
            scene_cut_music=sc.get("music", "stop"), scene_cut_cooldown=float(sc.get("cooldown", 10)),
            **autonomy_kwargs(cfg),
        )

        listen = cfg.raw.get("listen") or {}
        if asr is None:
            if opts.transcript:
                asr = ScriptedASR(load_transcript_file(opts.transcript))
            else:
                kind = opts.asr or listen.get("asr", "off")
                if opts.asr_model:
                    listen = {**listen, f"{kind}_model": opts.asr_model}
                asr = make_asr(kind, listen)
        self.asr = asr
        self.asr_worker = ASRWorker(asr, self.clock, lambda tr: None) if asr else None

        bcfg = cfg.raw.get("brain") or {}
        if brain is None:
            brain = make_brain(opts.brain or bcfg.get("provider", "off"), self.registry, bcfg)
        if brain is not None and self.fast and brain.name not in ("fake",):
            raise ValueError("облачная модель — только в реальном темпе (--speed 1): иначе задержки неверны")
        self.brain = brain
        self.scheduler = (BrainScheduler(brain, self.controller, self.journal, self.clock,
                                         SchedulerConfig.from_dict(bcfg)) if brain else None)

        scenecut = None
        if opts.video and self.info.has_video and (sc.get("mode", "SUGGEST") != "OFF" or opts.scene_cut_mode):
            sc_kind = {**sc, "kind": opts.scene_cut_kind} if opts.scene_cut_kind else sc
            scenecut = make_scenecut(sc_kind, lambda: self.controller.state.t_scene_start)
        vad_cfg = listen.get("vad") or {}
        self.listener = Listener(
            self.controller, self.journal, self.clock, None,
            vad=EnergyVAD(ASR_RATE, **vad_cfg), asr_worker=self.asr_worker,
            rules=FastRules(self.registry), scheduler=self.scheduler, scenecut=scenecut,
            synchronous=self.fast, threaded_cut=not self.fast,
        )
        self.feeder_error: str | None = None
        self.video_fps = float((cfg.raw.get("video") or {}).get("fps", 5))
        self.video_width = int((cfg.raw.get("video") or {}).get("width", 640))

    # ---- источники ---------------------------------------------------------------------
    def _audio(self) -> Iterator[tuple[float, str, Any]]:
        if not self.info.has_audio:
            return
        n = 0
        for block in read_audio_blocks(self.opts.media, ASR_RATE, AUDIO_BLOCK):
            t = n / ASR_RATE
            n += len(block)
            yield t + len(block) / ASR_RATE, "audio", (t, block)  # блок доступен после его конца

    def _video(self) -> Iterator[tuple[float, str, Any]]:
        if not (self.opts.video and self.info.has_video):
            return
        for t, rgb in read_video_frames(self.opts.media, self.video_fps, self.video_width):
            yield t, "frame", (t, rgb)

    def _events(self) -> Iterator[tuple[float, str, Any]]:
        merged = heapq.merge(self._audio(), self._video(), key=lambda e: e[0])
        for ev in merged:
            if self.opts.max_seconds and ev[0] > self.opts.max_seconds:
                break
            yield ev

    # ---- запуск ------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        return asyncio.run(self._run())

    async def _run(self) -> dict[str, Any]:
        if not self.fast and self.asr_worker:
            # Сначала загрузить модель: иначе начало записи пройдёт, пока она грузится,
            # а очередь держит лишь несколько фраз — ранняя речь потеряется.
            started = time.monotonic()
            self.asr_worker.start()
            ok = await asyncio.to_thread(self.asr_worker.wait_ready, ASR_WARMUP_TIMEOUT)
            self.journal.write("asr_warmup", state=self.asr_worker.state, ok=ok,
                               seconds=round(time.monotonic() - started, 2), error=self.asr_worker.last_error)
        t0 = self.clock.now()
        self.journal.write("replay_start", media_t0=round(t0, 4), speed=self.opts.speed,
                           asr=self.asr.name if self.asr else None,
                           brain=f"{self.brain.name}:{self.brain.model}" if self.brain else None,
                           duration=self.info.duration, has_video=self.info.has_video)
        self.controller.start_neutral_music()  # как вживую: первая сцена начинается с нейтральной
        if self.fast:
            last_t = await self._run_fast(t0)
        else:
            last_t = await self._run_realtime(t0)
        self.journal.write("replay_end", media_duration=round(last_t, 3))
        if self.brain:
            await self.brain.close()
        self.journal.close()
        from .report import load_journal

        records = load_journal(self.out_dir / "journal.jsonl")
        write_html(records, self.out_dir / "report.html", f"Прогон: {self.opts.media.name}")
        return summarize(records)

    async def _run_fast(self, t0: float) -> float:
        assert isinstance(self.clock, FakeClock)
        last_t = 0.0
        mix_rate = self.mixer.sample_rate
        for t_avail, kind, payload in self._events():
            if t_avail > last_t:
                frames = int((t_avail - last_t) * mix_rate)
                if frames > 0:
                    self.mixer.render(frames)
                self.clock.t = t0 + t_avail
                last_t = t_avail
            if kind == "audio":
                t, block = payload
                self.listener.feed_audio(block, t0 + t, ASR_RATE)
            else:
                t, rgb = payload
                self.listener.feed_frame(t0 + t, rgb)
            self.controller.tick()
            if self.scheduler:
                await self.scheduler.step()
        self.listener.flush_audio(t0 + last_t)
        self.controller.tick()
        if self.scheduler:
            await self.scheduler.step()
        return last_t

    async def _run_realtime(self, t0: float) -> float:
        loop = asyncio.get_running_loop()
        self.listener.loop = loop
        out = NullOutput(self.mixer, self.clock, on_block=self.engine.on_output_block)
        out.start()
        sched_task = asyncio.create_task(self.scheduler.run()) if self.scheduler else None
        done = threading.Event()
        last = {"t": 0.0}

        def feeder() -> None:
            try:
                for t_avail, kind, payload in self._events():
                    wait = t0 + t_avail - self.clock.now()
                    if wait > 0:
                        time.sleep(wait)
                    last["t"] = t_avail
                    if kind == "audio":
                        t, block = payload
                        self.listener.feed_audio(block, t0 + t, ASR_RATE)
                    else:
                        t, rgb = payload
                        self.listener.feed_frame(t0 + t, rgb)
                self.listener.flush_audio(t0 + last["t"])
            except Exception as e:  # noqa: BLE001 — сбой чтения/обработки: прогон завершается, а не висит
                self.feeder_error = f"{type(e).__name__}: {e}"[:500]
                self.journal.write("replay_error", where="feeder", error=self.feeder_error,
                                   media_t=round(last["t"], 3))
            finally:
                done.set()

        th = threading.Thread(target=feeder, name="replay-feeder", daemon=True)
        th.start()
        while not done.is_set():
            self.controller.tick()
            await asyncio.sleep(0.05)
        # Дождаться распознавания хвоста и последнего ответа модели.
        # Дождаться распознавания хвоста и последнего ответа модели. Готовый текст приходит в цикл
        # через call_soon_threadsafe, поэтому «пусто» должно повториться дважды подряд.
        deadline = time.monotonic() + 30
        idle_checks = 0
        while time.monotonic() < deadline and idle_checks < 2:
            self.controller.tick()
            idle_checks = 0 if self._draining_busy() else idle_checks + 1
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
        self.controller.tick()
        if sched_task:
            sched_task.cancel()
        if self.asr_worker:
            await asyncio.to_thread(self.asr_worker.stop)  # join не на потоке цикла
        if self.listener.cut_worker:
            await asyncio.to_thread(self.listener.cut_worker.stop)
        out.stop()
        return last["t"]

    def _draining_busy(self) -> bool:
        if self.asr_worker and self.asr_worker.busy:
            return True
        sc = self.scheduler
        # pending ждёт min_interval (should_request тогда None) — это тоже «ещё не всё».
        return bool(sc and sc.enabled and not sc.stopped_by_budget and (sc.in_flight or sc.pending))


def print_summary(s: dict[str, Any], out_dir: Path) -> None:
    print(f"\nГотово. Отчёт: {out_dir / 'report.html'}\nЖурнал: {out_dir / 'journal.jsonl'}\n")
    for k, v in s.items():
        print(f"  {k}: {v}")
