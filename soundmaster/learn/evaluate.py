"""Проверка детектора «монтажа» по разметке: модель учится на всех видео, кроме одного, и проверяется на нём.

    python -m soundmaster eval-scenecuts [--threshold 0.6 0.65] [--test max|auto|own] [--curve]

Так считаются цифры в docs/scenecut-plan.md — один и тот же способ для всех шагов плана, чтобы новый
признак сравнивался с прошлым результатом честно. Модель — случайный лес (scikit-learn) по признакам
траектории человека, который прошёл перед сценой; признаки считаются заново по рамкам из сбора.

Обучение — по человеку, которого указал Иван (tracks.json; в ~40% монтажей программа среагировала
не на того). На проверке подсказки нет, как и на шоу: модель оценивает каждого, кто заметно
сдвинулся в окне кандидата, и берёт самую высокую уверенность («max»). Для сравнения есть «auto» —
только тот, на кого среагировала программа, и «own» — человек Ивана (завышает результат: на шоу
такого нет; так были посчитаны первые цифры плана).

Полнота считается и «от начала до конца»: монтажи, которые отбор вообще не показал (ручные отметки M),
тоже пропуски. Ложные тревоги — и в штуках, и в час записи.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .collect import AFTER, BEFORE, covers
from .label import SERVICE, load_labels, load_manual, load_tracks
from .passes import is_scenecut_candidate, pass_features

# Шоу 1 и 2 — короткая форма, другой формат монтажа: не размечаем и не учим (решение Ивана, 2026-10-27)
EXCLUDE = {"uyutjmBh-ns", "ut93p5RVE0A"}
# Признаки, известные сразу после прохода, — для раннего шага (просадка музыки); LATE ждут ещё AFTER секунд,
# чтобы увидеть, кто где стоит после прохода
EARLY = ("span", "speed", "height", "top", "bottom", "pixel", "occlusion", "straight", "edge", "seen", "people_before")
LATE = EARLY + ("people_after", "moved")
MIN_SPAN = 0.25  # на проверке оцениваются люди, прошедшие хотя бы такую долю ширины кадра
TESTS = ("max", "auto", "own")


@dataclass
class Sample:
    video: str
    cid: str
    t: float
    cut: bool
    forced: bool  # кандидат есть только из-за ручной отметки M — на шоу его бы не было
    own: dict[str, float]  # признаки человека, указанного Иваном (или выбранного программой, если не указан)
    auto: dict[str, float]  # признаки того, кого выбрала программа
    people: list[dict[str, float]] = field(default_factory=list)  # все, кто заметно сдвинулся (с auto)


@dataclass
class Video:
    video: str
    title: str
    hours: float
    samples: list[Sample] = field(default_factory=list)
    missed: int = 0  # ручные отметки M, по которым кандидата нет (видео ещё не пересобрано)


def track_features(track: int, sources: list[str], frames, t_start: float, t_end: float) -> dict[str, float]:
    """pass_features + вход от края кадра и насколько непрерывно человека видно во время прохода."""
    f = pass_features(track, sources, frames, t_start, t_end, BEFORE, AFTER)
    pts = [(t, b[track]) for t, b in frames if track in b]
    f["edge"], f["seen"] = 0.5, 0.0
    if pts:
        x1, _, x2, _ = pts[0][1]
        f["edge"] = float(max(0.0, min(x1, 1.0 - x2)))  # 0 — вошёл из-за края кадра
        during = [t for t, _ in frames if pts[0][0] <= t <= pts[-1][0]]
        f["seen"] = len(pts) / max(1, len(during))
    return f


def candidate_sample(c: dict, cut: bool, picked: str | None, video: str) -> Sample:
    fr = [(t, {int(k): tuple(b) for k, b in bx.items()}) for t, bx in c.get("frames", [])]
    feats = {k: track_features(k, c["sources"], fr, c["t_start"], c["t_end"]) for k in {k for _, b in fr for k in b}}
    auto = feats.get(c["track"]) or track_features(c["track"], c["sources"], fr, c["t_start"], c["t_end"])
    own = feats.get(int(picked), auto) if picked is not None else auto
    people = [f for k, f in feats.items() if k != c["track"] and f["span"] >= MIN_SPAN] + [auto]
    forced = "manual" in c["sources"] and not (c["track"] >= 0 and is_scenecut_candidate(c.get("features", auto)))
    return Sample(video, c["id"], c["t_start"], cut, forced, own, auto, people)


def load_dataset(data_dir: Path, exclude: set[str] = EXCLUDE) -> list[Video]:
    """Размеченные кандидаты по видео. Метки cut/cut_other — монтаж, no — нет; unsure и без метки — мимо."""
    labels, tracks, manual = load_labels(data_dir), load_tracks(data_dir), load_manual(data_dir)
    out = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name in SERVICE:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["video"] in exclude:
            continue
        v = Video(d["video"], d.get("title", ""), d.get("seconds", 0.0) / 3600)
        cands = d.get("candidates", [])
        for c in cands:
            lab = labels.get(c["id"])
            if lab in ("cut", "cut_other", "no"):
                v.samples.append(candidate_sample(c, lab != "no", tracks.get(c["id"]), v.video))
        v.missed = sum(not any(covers(c, tm) for c in cands) for tm in manual.get(v.video, []))
        if v.samples:
            out.append(v)
    return out


def _x(f: dict[str, float], names: tuple[str, ...]) -> list[float]:
    return [f[n] for n in names]


def fit(samples: list[Sample], names: tuple[str, ...] = LATE, trees: int = 600):
    from sklearn.ensemble import RandomForestClassifier  # type: ignore

    X = np.array([_x(s.own, names) for s in samples], dtype=float).reshape(-1, len(names))
    model = RandomForestClassifier(trees, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1)
    return model.fit(X, np.array([s.cut for s in samples], dtype=bool))


def predict(model, samples: list[Sample], names: tuple[str, ...] = LATE, test: str = "max") -> np.ndarray:
    """Уверенность «монтаж» для каждого кандидата: test — max (лучший из людей в окне), auto или own."""
    if not samples:
        return np.zeros(0)
    groups = [s.people if test == "max" else [s.auto if test == "auto" else s.own] for s in samples]
    p = model.predict_proba(np.array([_x(f, names) for g in groups for f in g], dtype=float))[:, 1]
    out, i = [], 0
    for g in groups:
        out.append(float(p[i:i + len(g)].max()))
        i += len(g)
    return np.array(out)


def leave_one_out(videos: list[Video], names: tuple[str, ...] = LATE, test: str = "max",
                  trees: int = 600) -> dict[str, np.ndarray]:
    """Уверенность модели для кандидатов каждого видео, когда это видео в обучение не попало."""
    return {v.video: predict(fit([s for o in videos if o.video != v.video for s in o.samples], names, trees),
                             v.samples, names, test) for v in videos}


@dataclass
class Score:
    found: int = 0  # монтажей найдено (среди тех, что отбор показал)
    shown: int = 0  # монтажей среди кандидатов (без тех, что есть только из-за M)
    missed: int = 0  # монтажей, которых отбор не показал (M)
    false: int = 0  # ложных тревог
    nots: int = 0  # «не монтажей» среди кандидатов
    hours: float = 0.0

    def add(self, o: "Score") -> None:
        for k in ("found", "shown", "missed", "false", "nots", "hours"):
            setattr(self, k, getattr(self, k) + getattr(o, k))

    @property
    def cuts(self) -> int:
        return self.shown + self.missed

    @property
    def precision(self) -> float:
        return self.found / max(1, self.found + self.false)


def score(v: Video, prob: np.ndarray, threshold: float) -> Score:
    s = Score(missed=v.missed, hours=v.hours)
    for smp, p in zip(v.samples, prob):
        if smp.forced:
            s.missed += smp.cut
        elif smp.cut:
            s.shown += 1
            s.found += bool(p >= threshold)
        else:
            s.nots += 1
            s.false += bool(p >= threshold)
    return s


def learning_curve(videos: list[Video], names: tuple[str, ...] = LATE, threshold: float = 0.5, test: str = "max",
                   max_combos: int = 6) -> list[tuple[int, Score]]:
    """Как результат зависит от числа видео в обучении: для каждого проверочного видео — до max_combos
    наборов из n других (по порядку, без случайности), итог суммируется."""
    rows = []
    for n in range(1, len(videos)):
        tot = Score()
        for v in videos:
            rest = [o for o in videos if o.video != v.video]
            for combo in itertools.islice(itertools.combinations(rest, n), max_combos):
                m = fit([s for o in combo for s in o.samples], names, trees=300)
                tot.add(score(v, predict(m, v.samples, names, test), threshold))
        rows.append((n, tot))
    return rows


def _line(name: str, s: Score) -> str:
    return (f"{name[:28]:28} {s.found:3}/{s.shown:<3} {s.found / max(1, s.shown):4.0%}  "
            f"{s.found:3}/{s.cuts:<3} {s.found / max(1, s.cuts):4.0%}  {s.false:3}/{s.nots:<3} "
            f"{s.false / max(1e-9, s.hours):4.1f}/ч  {s.precision:4.0%}")


def report(data_dir: Path, thresholds: tuple[float, ...] = (0.5, 0.6), test: str = "max", curve: bool = False,
           early: bool = True) -> str:
    videos = load_dataset(data_dir)
    if len(videos) < 2:
        return "Нужно хотя бы два размеченных видео."
    smp = [s for v in videos for s in v.samples]
    lines = [f"Видео: {len(videos)} ({sum(v.hours for v in videos):.1f} ч). Размечено: монтажей "
             f"{sum(s.cut for s in smp)}, «не монтажей» {sum(not s.cut for s in smp)}; монтажей, которых отбор "
             f"не показал (M): {sum(v.missed for v in videos) + sum(s.cut and s.forced for s in smp)}. Проверка: {test}",
             f"{'':28} {'среди кандидатов':>15}  {'от начала до конца':>15}  {'ложные':>14}  точность"]
    for names, title in [(LATE, "все признаки (решение через ~7 с после прохода)")] + (
            [(EARLY, "только сразу после прохода")] if early else []):
        probs = leave_one_out(videos, names, test)
        for th in thresholds:
            lines.append(f"— {title}, порог {th}:")
            tot = Score()
            for v in videos:
                s = score(v, probs[v.video], th)
                tot.add(s)
                lines.append("  " + _line(v.title, s))
            lines.append("  " + _line("ИТОГО", tot))
    if curve:
        lines.append(f"Сколько видео в обучении (порог {thresholds[0]}): находит среди кандидатов / точность")
        lines += [f"  {n}: {s.found / max(1, s.shown):.0%} / {s.precision:.0%}"
                  for n, s in learning_curve(videos, threshold=thresholds[0], test=test)]
    return "\n".join(lines)
