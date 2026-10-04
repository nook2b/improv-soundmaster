"""Слух и зрение системы: звук → фразы → текст → правила и модель; кадры → детектор «монтажа».

Звук и кадры приходят из любых потоков (вход, камера, прогон записи); всё, что меняет
состояние контроллера, выполняется в потоке asyncio-цикла (контроллер однопоточный).
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from .brain.scheduler import BrainScheduler
from .clock import Clock
from .controller import Controller
from .fastrules import FastRules
from .journal import Journal
from .media.asr import ASRWorker, Transcript
from .media.scenecut import SceneCutDetector, SceneCutEvent
from .media.vad import EnergyVAD, Utterance

ASR_RATE = 16000


def to_asr_rate(block: np.ndarray, sr: int) -> np.ndarray:
    if sr == ASR_RATE:
        return block.astype(np.float32, copy=False)
    ratio = sr / ASR_RATE
    if abs(ratio - round(ratio)) < 1e-6:
        r = int(round(ratio))
        n = len(block) // r * r
        return block[:n].reshape(-1, r).mean(axis=1).astype(np.float32)  # усреднение = простой ФНЧ
    from .audio.wavio import resample_linear

    return resample_linear(block[:, None], sr, ASR_RATE)[:, 0]


class Listener:
    def __init__(
        self,
        controller: Controller,
        journal: Journal,
        clock: Clock,
        loop: asyncio.AbstractEventLoop | None,
        *,
        vad: EnergyVAD | None = None,
        asr_worker: ASRWorker | None = None,
        rules: FastRules | None = None,
        scheduler: BrainScheduler | None = None,
        scenecut: SceneCutDetector | None = None,
        synchronous: bool = False,
    ):
        self.controller = controller
        self.journal = journal
        self.clock = clock
        self.loop = loop
        self.vad = vad or EnergyVAD(ASR_RATE)
        self.asr_worker = asr_worker
        self.rules = rules
        self.scheduler = scheduler
        self.scenecut = scenecut
        self.synchronous = synchronous  # прогон «быстрее реального времени»: всё в одном потоке
        self.recent: list[dict[str, Any]] = []  # последние реплики для пульта
        self.utterances = 0
        if asr_worker:
            asr_worker.on_transcript = self._from_asr_thread

    def _post(self, fn, *args) -> None:
        if self.synchronous or self.loop is None:
            fn(*args)
        else:
            self.loop.call_soon_threadsafe(fn, *args)

    # ---- звук --------------------------------------------------------------------------
    def feed_audio(self, block: np.ndarray, t: float, sample_rate: int) -> None:
        mono = to_asr_rate(block, sample_rate)
        for u in self.vad.push(mono, t):
            self._on_utterance(u)

    def flush_audio(self, t_end: float) -> None:
        for u in self.vad.flush(t_end):
            self._on_utterance(u)

    def _on_utterance(self, u: Utterance) -> None:
        self.utterances += 1
        # Журнал потокобезопасен — пишем сразу из потока звука.
        self.journal.write("utterance", t_start=round(u.t_start, 3), t_end=round(u.t_end, 3),
                           forced_cut=u.forced_cut)
        if not self.asr_worker:
            return
        if self.synchronous:
            tr = self.asr_worker.process(u)
            if tr:
                self.on_transcript(tr)
        else:
            self.asr_worker.submit(u)

    def _from_asr_thread(self, tr: Transcript) -> None:
        self._post(self.on_transcript, tr)

    def on_transcript(self, tr: Transcript) -> None:
        self.journal.write("transcript", text=tr.text, t_start=round(tr.t_start, 3), t_end=round(tr.t_end, 3),
                           t_ready=round(tr.t_ready, 3), asr_seconds=round(tr.asr_seconds, 3),
                           delay=round(tr.t_ready - tr.t_end, 3), forced_cut=tr.forced_cut,
                           scene_id=self.controller.state.scene_id)
        self.recent.append({"t": round(tr.t_end, 1), "text": tr.text})
        del self.recent[:-8]
        if self.rules:
            for hit in self.rules.analyze(tr.text):
                if hit.negated:
                    self.journal.write("rule_negated", rule=hit.kind, effect=hit.effect, word=hit.word)
                    continue
                if hit.kind == "arm" and hit.effect:
                    self.controller.arm(hit.effect, source="fast_layer", reason=f"слово «{hit.word}»",
                                        t_input_ref=tr.t_end)
                elif hit.kind == "fire":
                    self.controller.fire("keyword", word=hit.word, t_signal=tr.t_end)
                elif hit.kind == "cancel":
                    a = self.controller.state.armed
                    if a and a.source in ("fast_layer", "ai"):
                        self.controller._disarm("cancel_phrase")
        if self.scheduler:
            self.scheduler.on_transcript(tr)

    # ---- кадры -------------------------------------------------------------------------
    def feed_frame(self, t: float, rgb: np.ndarray) -> None:
        if self.scheduler:
            self.scheduler.on_frame(t, rgb)
        if self.scenecut:
            ev = self.scenecut.push(t, rgb)
            if ev:
                self._post(self._on_scene_cut, ev)

    def _on_scene_cut(self, ev: SceneCutEvent) -> None:
        self.journal.write("scene_cut_detected", t=round(ev.t, 3), t_start=round(ev.t_start, 3),
                           direction=ev.direction, travel=round(ev.travel, 2), height=round(ev.height, 2),
                           confidence=round(ev.confidence, 2))
        self.controller.scene_cut("video", ev.confidence, ev.t)
