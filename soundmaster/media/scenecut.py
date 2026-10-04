"""Детектор «монтажа»: актёр проходит перед сценой — начинается новая сцена.

По кадрам камеры ищется крупный (высокий в кадре — значит, близко к камере) движущийся объект,
который за 0,4–4 с проходит большую часть ширины кадра в одном направлении.
Отсекаются: смена света (меняется почти весь кадр сразу), движение на месте, мелкие фигуры
в глубине сцены. Пороги подбираются по записям (`python -m soundmaster replay ... --report`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SceneCutConfig:
    analysis_width: int = 96
    pixel_threshold: int = 25  # изменение яркости пикселя, 0–255
    min_motion: float = 0.02  # доля движущихся пикселей, чтобы считать кадр «с движением»
    global_change: float = 0.55  # больше — это свет/переключение камеры, а не проход
    column_fill: float = 0.30  # колонка «занята», если в ней движется столько её высоты
    min_height: float = 0.35  # средняя высота объекта (доля кадра): проход близко к камере
    min_travel: float = 0.55  # сколько ширины кадра нужно пройти
    min_duration: float = 0.4
    max_duration: float = 4.0
    max_gap: float = 0.5  # допустимый разрыв движения внутри прохода, с
    min_monotonic: float = 0.8  # |корреляция| положения и времени
    cooldown: float = 3.0
    roi_top: float = 0.0  # зона анализа по высоте (доли кадра)
    roi_bottom: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "SceneCutConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                setattr(c, k, type(getattr(c, k))(v))
        return c


@dataclass
class SceneCutEvent:
    t: float  # время конца прохода
    t_start: float
    direction: str  # "left_to_right" | "right_to_left"
    travel: float
    height: float
    confidence: float


@dataclass
class _Track:
    points: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, height)
    last_t: float = 0.0


def to_gray_small(rgb: np.ndarray, width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    step = max(1, w // width)
    small = rgb[::step, ::step].astype(np.float32)
    return small[..., 0] * 0.299 + small[..., 1] * 0.587 + small[..., 2] * 0.114


class SceneCutDetector:
    def __init__(self, cfg: SceneCutConfig | None = None):
        self.cfg = cfg or SceneCutConfig()
        self._prev: np.ndarray | None = None
        self._track = _Track()
        self._last_event_t = -1e9
        self.last_motion = 0.0

    def reset(self) -> None:
        self._prev = None
        self._track = _Track()

    def push(self, t: float, rgb: np.ndarray) -> SceneCutEvent | None:
        c = self.cfg
        g = to_gray_small(rgb, c.analysis_width)
        h = g.shape[0]
        g = g[int(h * c.roi_top): max(int(h * c.roi_top) + 1, int(h * c.roi_bottom))]
        prev, self._prev = self._prev, g
        if prev is None or prev.shape != g.shape:
            return None
        mask = np.abs(g - prev) > c.pixel_threshold
        motion = float(mask.mean())
        self.last_motion = motion
        if motion >= c.global_change:
            self._track = _Track()  # свет или камера: не проход
            return None
        event = None
        if motion >= c.min_motion:
            col = mask.mean(axis=0)
            busy = np.nonzero(col >= c.column_fill)[0]
            if len(busy):
                x = float(busy.mean()) / (mask.shape[1] - 1)
                height = float(col[busy].mean())
                if self._track.points and t - self._track.last_t > c.max_gap:
                    event = self._finish()
                    self._track = _Track()
                self._track.points.append((t, x, height))
                self._track.last_t = t
                ev = self._check(final=False)
                if ev:
                    return ev
                return event
        if self._track.points and t - self._track.last_t > c.max_gap:
            event = self._finish()
            self._track = _Track()
        return event

    def _finish(self) -> SceneCutEvent | None:
        return self._check(final=True)

    def _check(self, final: bool) -> SceneCutEvent | None:
        c = self.cfg
        pts = self._track.points
        if len(pts) < 3:
            return None
        ts = np.array([p[0] for p in pts])
        xs = np.array([p[1] for p in pts])
        hs = np.array([p[2] for p in pts])
        duration = float(ts[-1] - ts[0])
        travel = float(xs.max() - xs.min())
        if duration > c.max_duration:
            # Слишком долго — это хождение по сцене. Оставляем только последние max_duration секунд.
            keep = ts >= ts[-1] - c.max_duration
            self._track.points = [p for p, k in zip(pts, keep) if k]
            return None
        if duration < c.min_duration or travel < c.min_travel:
            return None
        if np.std(xs) < 1e-6 or np.std(ts) < 1e-6:
            return None
        r = float(np.corrcoef(ts, xs)[0, 1])
        height = float(hs.mean())
        if abs(r) < c.min_monotonic or height < c.min_height:
            return None
        if ts[-1] - self._last_event_t < c.cooldown:
            return None
        self._last_event_t = float(ts[-1])
        self._track = _Track()
        conf = min(1.0, 0.4 * abs(r) + 0.3 * min(1.0, travel / 0.9) + 0.3 * min(1.0, height / 0.7))
        return SceneCutEvent(float(ts[-1]), float(ts[0]), "left_to_right" if r > 0 else "right_to_left",
                             travel, height, conf)
