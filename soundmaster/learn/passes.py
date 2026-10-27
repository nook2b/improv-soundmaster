"""Кандидаты в «монтаж» по траекториям людей: кто-то крупный быстро прошёл поперёк кадра.

Пороги подобраны по разметке Ивана («Компот 9», 285 отметок, 2026-10-24): монтажник крупный (рамка
не меньше ~половины кадра), голова высоко (не зритель в первом ряду), проходит заметную часть ширины,
и после прохода актёры на сцене стоят иначе, чем до (правило Ивана: если остальные остались — не монтаж),
или проход видит и старый детектор по пикселям. Так остаются 15 монтажей из 16 и четверть «не монтажей».
Отличать остальное будет модель, обученная на разметке. Координаты — доли кадра (0–1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 — доли кадра


@dataclass
class PassConfig:
    min_travel: float = 0.40  # какую долю ширины пройти
    min_height: float = 0.45  # средняя высота фигуры (доля кадра): ближе к камере — крупнее
    max_top: float = 0.45  # верх рамки выше этого (доля кадра от верха): у зрителей в зале рамка только снизу
    max_window: float = 6.0  # за сколько секунд, не дольше
    min_window: float = 0.3
    track_cooldown: float = 4.0  # одна фигура — не чаще одного кандидата за столько секунд


@dataclass
class PassCandidate:
    t_start: float
    t_end: float
    track: int  # id траектории (-1 — найден детектором по пикселям)
    travel: float
    height: float
    sources: list[str] = field(default_factory=lambda: ["people"])


class PassFinder:
    """Получает рамки людей по кадрам (с id траекторий) и отдаёт кандидатов «прошёл поперёк кадра»."""

    def __init__(self, cfg: PassConfig | None = None):
        self.cfg = cfg or PassConfig()
        self._hist: dict[int, deque[tuple[float, float, float, float]]] = {}  # id → (t, центр x, высота, верх)
        self._cooldown: dict[int, float] = {}

    def push(self, t: float, boxes: dict[int, Box]) -> list[PassCandidate]:
        c = self.cfg
        out = []
        for tid, (x1, y1, x2, y2) in boxes.items():
            h = self._hist.setdefault(tid, deque())
            h.append((t, (x1 + x2) / 2, y2 - y1, y1))
            while h and t - h[0][0] > c.max_window:
                h.popleft()
            if t < self._cooldown.get(tid, -1e9) or len(h) < 3:
                continue
            cx = h[-1][1]
            # самая ранняя точка окна, от которой фигура ушла на min_travel
            start = next((p for p in h if abs(cx - p[1]) >= c.min_travel and t - p[0] >= c.min_window), None)
            if start is None:
                continue
            seg = [p for p in h if p[0] >= start[0]]
            height = sum(p[2] for p in seg) / len(seg)
            top = sum(p[3] for p in seg) / len(seg)
            if height < c.min_height or top > c.max_top:
                continue
            xs = [p[1] for p in seg]
            out.append(PassCandidate(start[0], t, tid, max(xs) - min(xs), height))
            self._cooldown[tid] = t + c.track_cooldown
        for tid in [k for k in self._hist if k not in boxes and t - self._hist[k][-1][0] > c.max_window]:
            del self._hist[tid]
        return out


class FastTracker:
    """Связывает рамки людей между кадрами в траектории, с учётом скорости.

    Обычные трекеры (ByteTrack) связывают рамки по перекрытию — при 5 кадрах в секунду человек,
    пробегающий перед сценой, сдвигается на 15–20% ширины за кадр, рамки не перекрываются,
    и проход рвётся на куски. Здесь — ближайшая рамка к предсказанному положению, жадно.
    """

    def __init__(self, max_jump: float = 0.28, max_age: float = 1.0):
        self.max_jump, self.max_age = max_jump, max_age
        self._tracks: dict[int, tuple[float, Box, float]] = {}  # id → (t, рамка, скорость центра x в 1/с)
        self._next = 1

    def update(self, t: float, dets: list[Box]) -> dict[int, Box]:
        for tid in [k for k, (lt, _, _) in self._tracks.items() if t - lt > self.max_age]:
            del self._tracks[tid]
        pairs = []
        for tid, (lt, b, vx) in self._tracks.items():
            dt = t - lt
            px, py, ph = (b[0] + b[2]) / 2 + vx * dt, (b[1] + b[3]) / 2, b[3] - b[1]
            for j, d in enumerate(dets):
                dx = abs((d[0] + d[2]) / 2 - px)
                dy = abs((d[1] + d[3]) / 2 - py)
                dh = abs((d[3] - d[1]) - ph) / max(ph, 1e-3)
                if dx <= self.max_jump * max(1.0, dt / 0.2) and dy < 0.25 and dh < 0.6:
                    pairs.append((dx + dy + 0.3 * dh, tid, j))
        out: dict[int, Box] = {}
        used_t, used_d = set(), set()
        for _cost, tid, j in sorted(pairs):
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            lt, b, vx = self._tracks[tid]
            d = dets[j]
            nvx = (((d[0] + d[2]) - (b[0] + b[2])) / 2) / max(t - lt, 1e-3)
            self._tracks[tid] = (t, d, 0.5 * vx + 0.5 * nvx)
            out[tid] = d
        for j, d in enumerate(dets):
            if j not in used_d:
                self._tracks[self._next] = (t, d, 0.0)
                out[self._next] = d
                self._next += 1
        return out


Frames = list[tuple[float, dict[int, Box]]]  # рамки людей по времени: (t, {id траектории: рамка})


def pass_end(track: int, frames: Frames, t_end: float, horizon: float = 4.0) -> float:
    """Когда проход закончился: фигура ушла из кадра или перестала заметно двигаться."""
    pts = [(t, (b[track][0] + b[track][2]) / 2) for t, b in frames if track in b and t_end - 1 <= t <= t_end + horizon]
    end = t_end
    for (t0, x0), (t1, x1) in zip(pts, pts[1:]):
        if abs(x1 - x0) / max(t1 - t0, 1e-3) < 0.05:
            break
        end = t1
    return end


def pass_features(track: int, sources: list[str], frames: Frames, t_start: float, t_end: float,
                  before: float = 3.0, after: float = 7.0) -> dict[str, float]:
    """Признаки прохода для отбора и для модели. frames — рамки от t_start−before до t_end+after."""
    pix = 1.0 if "pixel" in sources else 0.0
    pts = [(t, b[track]) for t, b in frames if track in b]
    f = {"span": 0.0, "speed": 0.0, "height": 0.0, "top": 1.0, "bottom": 1.0, "pixel": pix, "occlusion": 0.0,
         "people_before": 0.0, "people_after": 0.0, "moved": 0.0, "straight": 0.0,
         "edge_in": 0.5, "edge_out": 0.5, "dur": 0.0, "vis": 0.0, "net": 0.0}
    if len(pts) >= 3:
        ts = np.array([p[0] for p in pts])
        bx = np.array([p[1] for p in pts])
        cx = (bx[:, 0] + bx[:, 2]) / 2
        speed = 0.0
        for i in range(len(ts)):
            j = int(np.searchsorted(ts, ts[i] + 1.0))
            if j < len(ts):
                speed = max(speed, abs(cx[j] - cx[i]) / (ts[j] - ts[i]))
        f.update(span=float(cx.max() - cx.min()), speed=speed, height=float((bx[:, 3] - bx[:, 1]).mean()),
                 top=float(bx[:, 1].mean()), bottom=float(bx[:, 3].mean()),
                 straight=float(abs(np.corrcoef(ts, cx)[0, 1])) if np.std(cx) > 0 else 0.0,
                 # вошёл и вышел у края кадра (монтажник появляется из-за кулис), сколько секунд виден,
                 # насколько непрерывно (смазанный бегущий теряется реже, чем перекрытый актёрами)
                 edge_in=float(min(bx[0, 0], 1 - bx[0, 2])), edge_out=float(min(bx[-1, 0], 1 - bx[-1, 2])),
                 dur=float(ts[-1] - ts[0]), net=float(abs(cx[-1] - cx[0])),
                 vis=len(pts) / max(1, sum(1 for t, _ in frames if ts[0] <= t <= ts[-1])))

        def overlap(a: Box, b: Box) -> float:
            return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1e-6, min(a[2] - a[0], b[2] - b[0]))

        f["occlusion"] = float(np.mean([sum(overlap(b[track], o) > 0.5 for k, o in b.items() if k != track)
                                        for t, b in frames if track in b]))

    def actors(t0: float, t1: float) -> list[float]:
        """Центры актёров (крупные фигуры, кроме идущего) в кадре окна, где их видно больше всего."""
        best: list[float] = []
        for t, b in frames:
            if t0 <= t <= t1:
                xs = [(o[0] + o[2]) / 2 for k, o in b.items() if k != track and o[3] - o[1] > 0.3]
                if len(xs) > len(best):
                    best = xs
        return best

    pre = actors(t_start - before, t_start - before / 2)
    post = actors(t_end + after - 2.0, t_end + after)
    f["people_before"], f["people_after"] = float(len(pre)), float(len(post))
    if pre:
        # доля актёров «до», которых «после» нет на прежнем месте (ушли или перешли)
        f["moved"] = float(np.mean([min((abs(x - y) for y in post), default=1.0) > 0.1 for x in pre]))
    return f


def is_scenecut_candidate(f: dict[str, float], cfg: PassConfig | None = None) -> bool:
    """Стоит ли показывать проход на разметку: крупный, не зритель, далеко, и сцена после него другая."""
    c = cfg or PassConfig()
    return (f["height"] >= c.min_height and f["top"] <= c.max_top and f["span"] >= c.min_travel
            and (f["moved"] > 0 or f["pixel"] > 0))


def merge(cands: list[PassCandidate], gap: float = 3.0, max_span: float = 1e9) -> list[PassCandidate]:
    """Кандидаты, которые пересекаются или ближе gap секунд, — один момент (источники объединяются).
    Дольше max_span секунд момент не растёт (кто-то ходит по сцене — цепочка не склеивается в минуту)."""
    out: list[PassCandidate] = []
    for c in sorted(cands, key=lambda c: c.t_start):
        if out and c.t_start <= out[-1].t_end + gap and max(out[-1].t_end, c.t_end) - out[-1].t_start <= max_span:
            m = out[-1]
            # рамку человека предпочитаем детектору по пикселям; из двух людей — того, кто прошёл дальше
            if c.track >= 0 and (m.track < 0 or c.travel > m.travel):
                m.track, m.travel, m.height = c.track, c.travel, c.height
            m.t_start, m.t_end = min(m.t_start, c.t_start), max(m.t_end, c.t_end)
            m.sources = sorted(set(m.sources) | set(c.sources))
        else:
            out.append(PassCandidate(c.t_start, c.t_end, c.track, c.travel, c.height, list(c.sources)))
    return out


@dataclass
class LivePass:
    """Проход, о котором детектор на шоу ещё не решил окончательно."""

    c: PassCandidate
    fast_done: bool = False
    state: dict = field(default_factory=dict)  # что решил детектор (вероятность, действие, «время сцены»)


class LivePasses:
    """Проходы так, как их видит детектор на шоу: без заглядывания вперёд.

    Один и тот же код — и на шоу (`media/peoplecut.py`), и при обучении (`learn/model.py` прогоняет через него
    сохранённые рамки): признаки модели считаются одинаково. Соседние проходы склеиваются, пока по
    моменту не принято быстрое решение (через fast_after с после прохода); полное — через full_after с.
    """

    GAP = 3.0  # проходы ближе — один момент (как merge при сборе)
    MAX_SPAN = 12.0

    def __init__(self, fast_after: float, full_after: float, cfg: PassConfig | None = None):
        self.fast_after, self.full_after = fast_after, full_after
        self.finder = PassFinder(cfg)
        self.pending: list[LivePass] = []

    def boxes(self, t: float, boxes: dict[int, Box]) -> None:
        for c in self.finder.push(t, boxes):
            self.add(c)

    def add(self, c: PassCandidate) -> None:
        for p in self.pending:
            m = p.c
            if c.t_start > m.t_end + self.GAP:
                continue
            if not p.fast_done and max(m.t_end, c.t_end) - m.t_start <= self.MAX_SPAN:
                if c.track >= 0 and (m.track < 0 or c.travel > m.travel):
                    m.track, m.travel, m.height = c.track, c.travel, c.height
                m.t_start, m.t_end = min(m.t_start, c.t_start), max(m.t_end, c.t_end)
                m.sources = sorted(set(m.sources) | set(c.sources))
                return
            if p.fast_done:
                # тот же момент ещё длится после быстрого решения: источник — в него, время — нет
                m.sources = sorted(set(m.sources) | set(c.sources))
                return
        self.pending.append(LivePass(PassCandidate(c.t_start, c.t_end, c.track, c.travel, c.height, list(c.sources))))

    def ready(self, t: float) -> list[tuple[str, LivePass]]:
        """Моменты, по которым пора решать: ("fast", p) — один раз, затем ("full", p) — и он уходит."""
        out = []
        for p in list(self.pending):
            if not p.fast_done and t >= p.c.t_end + self.fast_after:
                p.fast_done = True
                out.append(("fast", p))
            elif p.fast_done and t >= p.c.t_end + self.full_after:
                self.pending.remove(p)
                out.append(("full", p))
        return out
