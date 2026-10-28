"""Детектор «монтажа» по людям в кадре: обученная на разметке Ивана модель решает, менять ли сцену.

Как при сборе примеров (`learn/collect.py`): YOLO находит людей, FastTracker ведёт траектории,
PassFinder находит «крупный человек прошёл поперёк кадра», пиксельный детектор с мягкими порогами
помечает тот же проход как «pixel». Дальше модель (`learn/model.py`) в два шага:

  1. как только проход распознан (человек прошёл ~40% ширины) — fast: уверена → «cut» (новая сцена сразу); сомневается → «dip»
     (музыка приглушается, сцена не меняется);
  2. через 3 с — full: видно, ушли ли актёры → после «dip» — «confirm» (новая сцена) или «reject»
     (музыка возвращается); первый шаг пропустил, а второй уверен — «late_cut»; после «cut», если
     проверка не согласна, — «doubt» (только в журнал: сцену уже не вернуть).

Если YOLO не загрузилась (нет ultralytics, весов, памяти), работает старый детектор по пикселям.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..learn.model import FAST_AFTER, FULL_AFTER, CutModel, Policy, features_at
from ..learn.passes import Box, LivePasses, PassCandidate, PassConfig, is_scenecut_candidate
from .scenecut import SceneCutConfig, SceneCutDetector

MAX_SPAN = LivePasses.MAX_SPAN
BEFORE_KEEP = 6.0  # сколько секунд рамок держать до начала прохода (признакам нужно 3 с)
# пиксельный детектор — только как признак «pixel» (как LOOSE_PIXEL при сборе примеров)
LOOSE_PIXEL = SceneCutConfig(min_height=0.2, min_travel=0.35, max_duration=6.0, min_monotonic=0.7, cooldown=2.0)


@dataclass
class PeopleCutConfig:
    detect_fps: float = 5.0  # как при сборе примеров
    cut: float = 0.6
    dip: float = 0.5
    confirm: float = 0.5
    yolo: str = "yolo11s.pt"
    model: str = "config/scenecut_model.json"

    @classmethod
    def from_dict(cls, d: dict | None) -> "PeopleCutConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, type(getattr(c, k))(v))
        return c

    @property
    def policy(self) -> Policy:
        return Policy(self.cut, self.dip, self.confirm)


@dataclass
class PeopleCutEvent:
    action: str  # cut | dip | confirm | reject | late_cut | doubt | none
    t: float  # когда решено (время кадра)
    t_start: float
    t_end: float
    p: float  # вероятность «монтажа» на этом шаге
    stage: str  # fast | full | pixel
    features: dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.p


class PeopleCutDetector:
    def __init__(self, model: CutModel, people: Callable[[np.ndarray, float], dict[int, Box]],
                 cfg: PeopleCutConfig | None = None, scene_start: Callable[[], float] = lambda: 0.0,
                 fallback: SceneCutConfig | None = None):
        self.model, self.people, self.cfg = model, people, cfg or PeopleCutConfig()
        self.scene_start = scene_start
        self.pass_cfg = PassConfig()
        self.passes = LivePasses(FAST_AFTER, FULL_AFTER, self.pass_cfg)
        self.pixel = SceneCutDetector(LOOSE_PIXEL)
        self.fallback_cfg = fallback
        self.fallback: SceneCutDetector | None = None  # старый детектор, если YOLO не работает
        self.error: str | None = None
        self.log: deque[tuple[float, dict[int, Box]]] = deque()
        self._last_detect = -1e9

    @property
    def kind(self) -> str:
        return "pixel" if self.fallback else "people"

    def push(self, t: float, rgb: np.ndarray) -> list[PeopleCutEvent]:
        if self.fallback:
            ev = self.fallback.push(t, rgb)
            return [PeopleCutEvent("cut", ev.t, ev.t_start, ev.t, ev.confidence, "pixel")] if ev else []
        ev = self.pixel.push(t, rgb)
        if ev:
            self.passes.add(PassCandidate(ev.t_start, ev.t, -1, ev.travel, ev.height, ["pixel"]))
        if t - self._last_detect >= 1.0 / self.cfg.detect_fps - 1e-3:
            self._last_detect = t
            try:
                boxes = self.people(rgb, t)
            except Exception as e:  # noqa: BLE001 — без нейросети «монтаж» всё равно ловится
                self.error = f"{type(e).__name__}: {e}"
                self.fallback = SceneCutDetector(self.fallback_cfg)
                return []
            self.boxes(t, boxes)
        return self.step(t)

    def boxes(self, t: float, boxes: dict[int, Box]) -> None:
        boxes = {k: b for k, b in boxes.items() if b[3] - b[1] > 0.12}  # как в собранных примерах
        self.log.append((t, boxes))
        keep = t - (BEFORE_KEEP + MAX_SPAN + FULL_AFTER)
        while self.log and self.log[0][0] < keep:
            self.log.popleft()
        self.passes.boxes(t, boxes)

    def step(self, t: float) -> list[PeopleCutEvent]:
        out = []
        pol = self.cfg.policy
        for stage, p in self.passes.ready(t):
            c, st = p.c, p.state
            if stage == "fast":
                st["age"] = c.t_start - self.scene_start()  # до нашей же смены сцены
                if c.track < 0:  # только пиксельный: при сборе такие не показывались — модель их не знает
                    st["skip"] = True
                    continue
                f = features_at(c.track, c.sources, self.log, c.t_start, c.t_end, FAST_AFTER, st["age"])
                st["p"] = self.model.proba("fast", f)
                st["action"] = "cut" if st["p"] >= pol.cut else "dip" if st["p"] >= pol.dip else None
                out.append(PeopleCutEvent(st["action"] or "none", t, c.t_start, c.t_end, st["p"], "fast", f))
            elif not st.get("skip"):
                f = features_at(c.track, c.sources, self.log, c.t_start, c.t_end, FULL_AFTER, st["age"])
                p_full = self.model.proba("full", f)
                ok = is_scenecut_candidate(f, self.pass_cfg)
                action = st["action"]
                if action == "dip":
                    action = "confirm" if ok and p_full >= pol.confirm else "reject"
                elif action == "cut":
                    action = "doubt" if not ok or p_full < pol.confirm else "none"
                else:
                    action = "late_cut" if ok and p_full >= pol.cut else "none"
                if action != "none" or st["action"]:
                    out.append(PeopleCutEvent(action, t, c.t_start, c.t_end, p_full, "full", f))
        return out


class FrameWorker:
    """Нейросеть на кадрах — в своём потоке: камера и цикл пульта не ждут YOLO.

    Если поток не успевает, старые кадры выбрасываются (траектории ведутся по времени кадра).
    """

    def __init__(self, detector: PeopleCutDetector, on_events: Callable[[list[PeopleCutEvent]], None]):
        self.detector, self.on_events = detector, on_events
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def submit(self, t: float, rgb: np.ndarray) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="peoplecut", daemon=True)
            self._thread.start()
        try:
            self._q.put_nowait((t, rgb))
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self._q.put_nowait((t, rgb))

    def stop(self) -> None:
        if self._thread:
            self._q.put(None)
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            evs = self.detector.push(*item)
            if evs:
                self.on_events(evs)


class LazyYolo:
    """YOLO грузится при первом кадре (в потоке FrameWorker), а не при запуске пульта."""

    def __init__(self, model: str):
        self.model = model
        self._impl = None

    def __call__(self, rgb: np.ndarray, t: float) -> dict[int, Box]:
        if self._impl is None:
            from ..learn.collect import YoloPeople

            self._impl = YoloPeople(self.model)
        return self._impl(rgb, t)


def make_scenecut(sc: dict, scene_start: Callable[[], float]):
    """Детектор из раздела scene_cut show.yaml: kind people (модель) или pixel (старый)."""
    from pathlib import Path

    pixel_cfg = SceneCutConfig.from_dict(sc.get("detector"))
    if str(sc.get("kind", "people")) == "people":
        pc = PeopleCutConfig.from_dict(sc.get("people"))
        path = Path(pc.model)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parents[2] / pc.model  # путь от корня проекта, а не от папки запуска
        if path.exists():
            return PeopleCutDetector(CutModel.load(path), LazyYolo(pc.yolo), pc, scene_start, pixel_cfg)
    return SceneCutDetector(pixel_cfg)
