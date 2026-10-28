"""Модель «монтаж / не монтаж» по признакам прохода: обучение на разметке Ивана и работа на шоу.

Две ступени (решение 2026-10-28, `docs/scenecut-plan.md`, шаг 6):
  - fast — через FAST_AFTER с после прохода: решает, менять ли сцену сразу или только приглушить
    музыку и подождать;
  - full — через FULL_AFTER с: видно, расходятся ли актёры (правило Ивана);
    подтверждает или отменяет приглушение.

Обучение — случайный лес sklearn (`pip install scikit-learn`, только для `train-scenecut`).
На шоу sklearn не нужен: деревья сохраняются в JSON и считаются здесь же, на чистом Python.

    python -m soundmaster train-scenecut          # data/scenecut → config/scenecut_model.json
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .passes import Frames, LivePass, LivePasses, pass_features

# 0 с: решение — в момент, когда проход распознан; проверено 2026-10-28 на 0 / 0,5 / 1 с — без ожидания
# точность не хуже (81% против 78%), а сразу решается больше монтажей (Иван: главное — быстродействие)
FAST_AFTER = 0.0
# 3, а не 7 с (= collect.AFTER, сколько кадров хранится): проверено 2026-10-28 на 2–7 с — точность та же,
# а сомнительный монтаж подтверждается на 4 с раньше (Иван: ждать долго — странно)
FULL_AFTER = 3.0
BEFORE = 3.0
MAX_AGE = 600.0  # «время сцены» дольше 10 минут не различаем
# признаки по порядку: проход (pass_features) + сколько секунд идёт текущая сцена
FEATURES = ("dur", "edge_in", "edge_out", "height", "moved", "net", "occlusion", "people_after", "people_before",
            "pixel", "span", "speed", "straight", "top", "bottom", "vis", "scene_age")
SKIP_FILES = {"labels.json", "manual.json", "tracks.json", "ignore.json"}


@dataclass
class Forest:
    """Деревья решений: в каждом узле признак, порог, левый/правый потомок; в листе — доля «монтажа»."""

    trees: list[dict] = field(default_factory=list)

    def predict(self, x: Sequence[float]) -> float:
        total = 0.0
        for tr in self.trees:
            feat, thr, left, right, value = tr["f"], tr["t"], tr["l"], tr["r"], tr["v"]
            i = 0
            while left[i] >= 0:
                i = left[i] if x[feat[i]] <= thr[i] else right[i]
            total += value[i]
        return total / max(1, len(self.trees))

    @classmethod
    def from_sklearn(cls, model) -> "Forest":
        trees = []
        for est in model.estimators_:
            t = est.tree_
            v = t.value[:, 0, :]
            p1 = (v[:, 1] / v.sum(axis=1).clip(min=1e-12)).round(4)
            trees.append({"f": [int(f) for f in t.feature], "t": [round(float(x), 5) for x in t.threshold],
                          "l": [int(x) for x in t.children_left], "r": [int(x) for x in t.children_right],
                          "v": [float(x) for x in p1]})
        return cls(trees)


@dataclass
class CutModel:
    stages: dict[str, Forest]
    features: tuple[str, ...] = FEATURES
    info: dict = field(default_factory=dict)  # когда обучена, на чём, проверка

    def proba(self, stage: str, feats: dict[str, float]) -> float:
        return self.stages[stage].predict([float(feats.get(k, 0.0)) for k in self.features])

    @classmethod
    def load(cls, path: Path) -> "CutModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: Forest(v) for k, v in d["stages"].items()}, tuple(d["features"]), d.get("info", {}))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"features": list(self.features), "info": self.info,
                                    "stages": {k: f.trees for k, f in self.stages.items()}},
                                   ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def features_at(track: int, sources: list[str], frames: Frames, t_start: float, t_end: float, after: float,
                scene_age: float) -> dict[str, float]:
    """Признаки так, как их видит детектор на шоу через after с после прохода (будущих кадров нет)."""
    window = [(t, b) for t, b in frames if t_start - BEFORE <= t <= t_end + after]
    f = pass_features(track, sources, window, t_start, t_end, BEFORE, after)
    f["scene_age"] = min(MAX_AGE, max(0.0, scene_age))
    return f


@dataclass
class Example:
    video: str
    title: str
    t: float
    y: bool
    fast: dict[str, float]
    full: dict[str, float]


def live_features(frames: Frames, pixel: bool, scene_age: float, near: tuple[float, float],
                  track: int | None = None) -> tuple[dict[str, float], dict[str, float]] | None:
    """Прогнать сохранённые рамки через тот же поиск проходов, что на шоу (LivePasses), и взять признаки
    момента рядом с near=(начало, конец): (fast, full). None — на шоу этот момент не нашёлся бы.

    pixel — при сборе этот момент видел и пиксельный детектор (на шоу он помечает проход так же).
    track — человек, указанный Иваном вручную (B): если на шоу нашёлся и его проход, берём его.
    """
    lp = LivePasses(FAST_AFTER, FULL_AFTER)
    seen: list[LivePass] = []
    for t, b in frames:
        lp.boxes(t, b)
        for stage, p in lp.ready(t):
            if stage == "fast":
                seen.append(p)
    for p in lp.pending:
        if not p.fast_done:
            seen.append(p)
    hits = [p for p in seen if p.c.track >= 0 and p.c.t_start <= near[1] + 1.0 and p.c.t_end >= near[0] - 1.0]
    if not hits:
        return None
    hit = next((p for p in hits if p.c.track == track), hits[0])
    c = hit.c
    sources = sorted(set(c.sources) | ({"pixel"} if pixel else set()))
    return tuple(features_at(c.track, sources, frames, c.t_start, c.t_end, after, scene_age)  # type: ignore[return-value]
                 for after in (FAST_AFTER, FULL_AFTER))


def examples(data_dir: Path, progress: Callable[[str], None] | None = None) -> list[Example]:
    """Размеченные кандидаты (cut / cut_other / no) с признаками — такими, какими их увидел бы детектор на шоу."""
    from .label import ignored, load_ignore, load_labels, load_manual, load_tracks

    labels, tracks, manual = load_labels(data_dir), load_tracks(data_dir), load_manual(data_dir)
    skip = load_ignore(data_dir)
    out, lost = [], 0
    for p in sorted(data_dir.glob("*.json")):
        if p.name in SKIP_FILES:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        vid = d.get("video", p.stem)
        off = skip.get(vid, [])
        cands = [c for c in d.get("candidates", []) if not ignored(off, c["t_start"])]
        # начала сцен — размеченные монтажи, ручные отметки и концы пропущенных отрезков (разминка
        # кончилась — пошли сцены): «время сцены» как на шоу
        cuts = sorted([c["t_start"] for c in cands if labels.get(c["id"]) in ("cut", "cut_other")]
                      + [t for t in manual.get(vid, []) if not ignored(off, t)] + [b for _, b in off])
        for c in cands:
            lab = labels.get(c["id"])
            if lab not in ("cut", "cut_other", "no"):
                continue
            frames = [(t, {int(k): tuple(v) for k, v in b.items()}) for t, b in c["frames"]]
            prev = [x for x in cuts if x < c["t_start"] - 8]
            age = c["t_start"] - (prev[-1] if prev else 0.0)
            tr = int(tracks[c["id"]]) if c["id"] in tracks else None
            f = live_features(frames, "pixel" in c["sources"], age, (c["t_start"], c.get("t_pass_end", c["t_end"])), tr)
            if f is None:
                lost += lab != "no"
                continue
            out.append(Example(vid, d.get("title", vid), c["t_start"], lab != "no", f[0], f[1]))
    if progress and lost:
        progress(f"  монтажей, которые детектор на шоу не нашёл бы как проход: {lost} (в обучение не идут)")
    return out


@dataclass
class Policy:
    """Как детектор на шоу превращает вероятности в действия (пороги — PeopleCutConfig)."""

    # подобраны по разметке 2026-10-28 (7 шоу, каждое — моделью, которая его не видела, признаки — как
    # на шоу, LivePasses): находит 80% монтажей, точность 80%, зря приглушает музыку ~1–2 раза за шоу
    cut: float = 0.6  # fast ≥ — сразу новая сцена
    dip: float = 0.5  # fast ≥ — приглушить музыку и ждать проверки full
    confirm: float = 0.5  # full ≥ (и актёры сдвинулись или видит пиксельный) — новая сцена


def decide(policy: Policy, p_fast: float, p_full: float, full_ok: bool) -> tuple[bool, bool]:
    """(сменили ли сцену, приглушали ли музыку зря)."""
    if p_fast >= policy.cut:
        return True, False
    if p_fast >= policy.dip:
        ok = full_ok and p_full >= policy.confirm
        return ok, not ok
    return full_ok and p_full >= policy.cut, False  # быстрая ступень пропустила, полная уверена — позже


def train(data_dir: Path, out: Path, trees: int = 200, policy: Policy | None = None,
          progress: Callable[[str], None] = print) -> CutModel:
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:  # pragma: no cover — зависит от окружения
        raise RuntimeError("для обучения нужен scikit-learn: pip install scikit-learn") from e
    from .passes import is_scenecut_candidate

    policy = policy or Policy()
    ex = examples(data_dir, progress)
    if len({e.video for e in ex}) < 2 or not any(e.y for e in ex) or all(e.y for e in ex):
        raise RuntimeError("мало разметки: нужны хотя бы два видео с монтажами и «не монтажами»")

    def fit(rows: list[Example], stage: str):
        X = np.array([[getattr(e, stage)[k] for k in FEATURES] for e in rows])
        y = np.array([e.y for e in rows])
        return RandomForestClassifier(trees, min_samples_leaf=2, class_weight="balanced", random_state=0,
                                      n_jobs=-1).fit(X, y)

    def proba(m, e: Example, stage: str) -> float:
        return float(m.predict_proba(np.array([[getattr(e, stage)[k] for k in FEATURES]]))[0, 1])

    # проверка: каждое видео — моделью, которая его не видела
    by_video: dict[str, list[Example]] = defaultdict(list)
    for e in ex:
        by_video[e.video].append(e)
    per_video, tot = [], defaultdict(int)
    for vid, test in by_video.items():
        rest = [e for e in ex if e.video != vid]
        mf, mF = fit(rest, "fast"), fit(rest, "full")
        r = defaultdict(int)
        for e in test:
            full_ok = is_scenecut_candidate(e.full)
            cut, wasted = decide(policy, proba(mf, e, "fast"), proba(mF, e, "full"), full_ok)
            r["cuts" if e.y else "no"] += 1
            r["found" if e.y and cut else "false" if cut else "wasted_dip" if wasted else "_"] += 1
        per_video.append({"video": vid, "title": test[0].title, **{k: v for k, v in r.items() if k != "_"}})
        for k, v in r.items():
            tot[k] += v
    found, false = tot["found"], tot["false"]
    check = {"cuts": tot["cuts"], "no": tot["no"], "found": found, "false": false, "wasted_dip": tot["wasted_dip"],
             "recall": round(found / max(1, tot["cuts"]), 3), "precision": round(found / max(1, found + false), 3)}
    for v in per_video:
        progress(f"  {v['title'][:32]:32s} нашла {v.get('found', 0)}/{v.get('cuts', 0)}, "
                 f"ложных {v.get('false', 0)}/{v.get('no', 0)}, зря приглушила {v.get('wasted_dip', 0)}")
    progress(f"Итого (каждое видео — моделью, которая его не видела): нашла {found} из {tot['cuts']} "
             f"({check['recall']:.0%}), ложных смен сцены {false} из {tot['no']}, точность {check['precision']:.0%}, "
             f"зря приглушила музыку {tot['wasted_dip']}")
    info = {"trained": time.strftime("%Y-%m-%d %H:%M"), "examples": len(ex), "videos": len(by_video),
            "policy": policy.__dict__, "check": check, "per_video": per_video}
    model = CutModel({"fast": Forest.from_sklearn(fit(ex, "fast")), "full": Forest.from_sklearn(fit(ex, "full"))},
                     FEATURES, info)
    model.save(out)
    progress(f"Модель записана: {out} ({out.stat().st_size // 1024} КБ)")
    return model
