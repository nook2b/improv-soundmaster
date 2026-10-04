"""Когда и что отправлять облачной модели; разбор её ответов в команды контроллера.

Запрос уходит после новой реплики (не чаще min_interval) или, если долго тихо, с одним кадром
(жесты без слов). Одновременно — не больше одного запроса: устаревшие ответы всё равно
отбросит контроллер. Каждое решение привязано к сцене, соединению и времени реплики.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..clock import Clock
from ..controller import Controller
from ..journal import Journal
from ..media.asr import Transcript
from .providers import Brain, encode_jpeg


@dataclass
class SchedulerConfig:
    min_interval: float = 1.5
    idle_interval: float = 10.0
    image_every: float = 5.0  # не чаще, с; 0 — без кадров
    context_seconds: float = 45.0
    budget_usd: float = 3.0
    jpeg_quality: int = 70

    @classmethod
    def from_dict(cls, d: dict | None) -> "SchedulerConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, type(getattr(c, k))(v))
        return c


def fmt_t(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


class BrainScheduler:
    def __init__(self, brain: Brain, controller: Controller, journal: Journal, clock: Clock,
                 cfg: SchedulerConfig | None = None):
        self.brain = brain
        self.controller = controller
        self.journal = journal
        self.clock = clock
        self.cfg = cfg or SchedulerConfig()
        self.transcripts: deque[Transcript] = deque(maxlen=200)
        self.sent_upto = 0  # сколько реплик сцены уже ушло модели
        self.note = ""
        self.pending = False
        self.in_flight = False
        self.last_request_t = -1e9
        self.last_image_t = -1e9
        self.last_activity_t = clock.now()
        self.frame: tuple[float, np.ndarray] | None = None
        self.cost = 0.0
        self.requests = 0
        self.errors = 0
        self.latencies: list[float] = []
        self.stopped_by_budget = False
        self.enabled = True
        self._scene_lines = 0
        controller.new_scene_listeners.append(self.reset_scene)

    # ---- входы -------------------------------------------------------------------------
    def on_transcript(self, tr: Transcript) -> None:
        self.transcripts.append(tr)
        self._scene_lines += 1
        self.pending = True
        self.last_activity_t = self.clock.now()

    def on_frame(self, t: float, rgb: np.ndarray) -> None:
        self.frame = (t, rgb)

    def reset_scene(self, scene_id: int, source: str) -> None:
        self.transcripts.clear()
        self._scene_lines = 0
        self.sent_upto = 0
        self.note = ""
        self.pending = False
        self.last_activity_t = self.clock.now()

    # ---- расписание --------------------------------------------------------------------
    def should_request(self) -> str | None:
        if not self.enabled or self.in_flight or self.stopped_by_budget:
            return None
        now = self.clock.now()
        if self.pending and now - self.last_request_t >= self.cfg.min_interval:
            return "transcript"
        if (self.frame is not None and self.cfg.image_every > 0
                and now - max(self.last_request_t, self.last_activity_t) >= self.cfg.idle_interval):
            return "idle_frame"
        return None

    async def step(self) -> bool:
        reason = self.should_request()
        if reason:
            await self.request(reason)
            return True
        return False

    async def run(self, period: float = 0.05) -> None:
        while True:
            try:
                if not await self.step():
                    await asyncio.sleep(period)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — цикл модели не должен ронять пульт
                self.journal.write("error", where="brain_scheduler", error=f"{type(e).__name__}: {e}")
                await asyncio.sleep(1.0)

    # ---- контекст ----------------------------------------------------------------------
    def build_context(self) -> tuple[str, float | None]:
        c = self.controller
        now = self.clock.now()
        t0 = c.state.t_scene_start
        lines = [f"СОСТОЯНИЕ: сцена {c.state.scene_id}, идёт {fmt_t(now - t0)}."]
        if c.state.scene_hint:
            lines.append(f"Подсказка зала: «{c.state.scene_hint}».")
        if self.note:
            lines.append(f"Твоя заметка о сцене: {self.note}")
        playing = sorted({e for _, e, _, _ in c.engine.mixer.playing})
        lines.append("Сейчас играет: " + (", ".join(playing) if playing else "ничего") + ".")
        if c.state.armed:
            lines.append(f"Взведено: {c.state.armed.effect}.")
        recent = [p for p in c.played if now - p.t < 60]
        if recent:
            lines.append("Недавно звучало: " + "; ".join(f"{p.effect} ({int(now - p.t)} с назад)" for p in recent[-5:]))
        dismissed = [k for k, t in c.dismissed.items() if now - t < 60]
        if dismissed:
            lines.append("Техник отклонил: " + ", ".join(dismissed))

        window = [tr for tr in self.transcripts if now - tr.t_end <= self.cfg.context_seconds]
        n_new = max(0, min(self._scene_lines - self.sent_upto, len(window)))
        old, new = window[: len(window) - n_new], window[len(window) - n_new:]
        lines.append("\nРЕПЛИКИ ДО ЭТОГО:" if old else "\nРЕПЛИК ДО ЭТОГО НЕТ.")
        lines += [f"[{fmt_t(tr.t_start - t0)}] {tr.text}" for tr in old]
        lines.append("\nНОВОЕ:")
        lines += [f"[{fmt_t(tr.t_start - t0)}] {tr.text}" for tr in new] or ["(новых реплик нет)"]
        t_ref = new[-1].t_end if new else None
        return "\n".join(lines), t_ref

    # ---- запрос ------------------------------------------------------------------------
    async def request(self, reason: str) -> None:
        c = self.controller
        now = self.clock.now()
        self.in_flight = True
        self.pending = False
        self.last_request_t = now
        scene_id, conn_id = c.state.scene_id, c.state.connection_id
        context, t_ref = self.build_context()
        self.sent_upto = self._scene_lines
        image = None
        if self.frame is not None and self.cfg.image_every > 0 and now - self.last_image_t >= self.cfg.image_every:
            t_frame, rgb = self.frame
            image = encode_jpeg(rgb, self.cfg.jpeg_quality)
            self.last_image_t = now
            if t_ref is None:
                t_ref = t_frame
        if t_ref is None:
            t_ref = now
        self.requests += 1
        rid = f"b{self.requests}"
        self.journal.write("brain_request", request_id=rid, reason=reason, scene_id=scene_id,
                           t_input_ref=round(t_ref, 3), image=bool(image), context_chars=len(context),
                           context=context)
        try:
            res = await self.brain.decide(context, image)
        finally:
            self.in_flight = False
        self.cost += res.cost
        self.latencies.append(res.latency)
        if res.error:
            self.errors += 1
        self.journal.write("brain_response", request_id=rid, latency=round(res.latency, 3), actions=res.actions,
                           error=res.error, cost=round(res.cost, 6), cost_total=round(self.cost, 4),
                           input_tokens=res.usage.input_tokens, cached_tokens=res.usage.cached_tokens,
                           output_tokens=res.usage.output_tokens, reasoning_tokens=res.usage.reasoning_tokens)
        if self.cost >= self.cfg.budget_usd:
            self.stopped_by_budget = True
            self.journal.write("budget_exceeded", cost_total=round(self.cost, 4), budget=self.cfg.budget_usd)
        for a in res.actions:
            self.apply(a, scene_id, conn_id, t_ref, rid)

    def apply(self, a: dict[str, Any], scene_id: int, conn_id: int, t_ref: float, rid: str) -> None:
        c = self.controller
        kw = dict(scene_id=scene_id, connection_id=conn_id, request_id=rid,
                  reason=a.get("reason", ""), confidence=a.get("confidence"))
        typ, target = a.get("type"), a.get("id")
        if typ == "note":
            if scene_id == c.state.scene_id:
                self.note = a.get("reason", "")[:200]
            return
        if typ == "scene_end":
            c.ai_scene_end(**kw)
            return
        if not target:
            self.journal.write("brain_invalid", request_id=rid, action=a, why="нет id")
            return
        if typ == "effect":
            c.ai_effect(target, t_input_ref=t_ref, call="suggest_sfx", **kw)
        elif typ == "arm":
            c.arm(target, source="ai", t_input_ref=t_ref, scene_id=scene_id, connection_id=conn_id,
                  reason=kw["reason"], request_id=rid)
        elif typ == "background":
            c.ai_music(target, t_input_ref=t_ref, **kw)
        else:
            self.journal.write("brain_invalid", request_id=rid, action=a, why="неизвестный тип")

    def status(self) -> dict[str, Any]:
        lat = sorted(self.latencies)
        return {
            "provider": self.brain.name, "model": self.brain.model, "requests": self.requests,
            "errors": self.errors, "cost": round(self.cost, 4), "budget": self.cfg.budget_usd,
            "stopped_by_budget": self.stopped_by_budget, "enabled": self.enabled, "note": self.note,
            "latency_p50": round(lat[len(lat) // 2], 2) if lat else None,
        }
