"""Детектор «монтажа» по людям в кадре: модель в два шага (приглушить → подтвердить/отменить)."""

import numpy as np
import pytest

from soundmaster.learn.model import CutModel, Forest
from soundmaster.media.peoplecut import PeopleCutConfig, PeopleCutDetector, make_scenecut
from soundmaster.media.scenecut import SceneCutDetector


def const_model(fast: float, full: float) -> CutModel:
    leaf = lambda p: Forest([{"f": [-2], "t": [-2.0], "l": [-1], "r": [-1], "v": [p]}])  # noqa: E731
    return CutModel({"fast": leaf(fast), "full": leaf(full)})


def scene(actors_leave: bool):
    """Двое стоят; на 5–6,2 с кто-то крупный пробегает через весь кадр; потом актёры ушли или остались."""
    def people(rgb, t):
        b = {}
        if t < 6.5 or not actors_leave:
            b = {1: (0.24, 0.25, 0.36, 0.9), 2: (0.54, 0.25, 0.66, 0.9)}
        if 5.0 <= t <= 6.2:
            x = 0.05 + 0.75 * (t - 5.0)
            b[9] = (x - 0.08, 0.15, x + 0.08, 0.95)
        return b
    return people


def run(det, seconds=16.0, fps=10):
    rgb = np.zeros((36, 64, 3), np.uint8)
    out = []
    for i in range(int(seconds * fps)):
        out += det.push(i / fps, rgb)
    return out


@pytest.mark.parametrize("fast,full,leave,expected", [
    (0.55, 0.9, True, ["dip", "confirm"]),  # сомневается → приглушила → актёры ушли → новая сцена
    (0.55, 0.9, False, ["dip", "reject"]),  # актёры остались (правило Ивана) → музыка возвращается
    (0.55, 0.2, True, ["dip", "reject"]),  # проверка не уверена
    (0.9, 0.9, True, ["cut", "none"]),  # уверена сразу
    (0.9, 0.9, False, ["cut", "doubt"]),  # сменила сразу, а актёры остались — в журнал
    (0.1, 0.9, True, ["none", "late_cut"]),  # первый шаг пропустил, второй уверен
    (0.1, 0.2, True, ["none"]),
])
def test_two_step_decisions(fast, full, leave, expected):
    det = PeopleCutDetector(const_model(fast, full), scene(leave), PeopleCutConfig())
    evs = run(det)
    assert [e.action for e in evs] == expected
    assert evs[0].stage == "fast" and 6.0 <= evs[0].t <= 8.0  # ~1 с после прохода
    if len(evs) > 1:
        assert evs[1].stage == "full" and evs[1].t >= evs[0].t + 1.5


def test_scene_age_feature_from_scene_start():
    det = PeopleCutDetector(const_model(0.9, 0.9), scene(True), PeopleCutConfig(), scene_start=lambda: 2.0)
    evs = run(det)
    assert evs[0].features["scene_age"] == pytest.approx(evs[0].t_start - 2.0)


def test_yolo_failure_falls_back_to_pixels():
    def broken(rgb, t):
        raise ImportError("No module named 'ultralytics'")

    det = PeopleCutDetector(const_model(0.9, 0.9), broken, PeopleCutConfig())
    assert run(det, seconds=1) == [] and det.kind == "pixel" and "ultralytics" in det.error


def test_forest_matches_sklearn():
    sk = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(0)
    X = rng.random((200, 5))
    y = X[:, 0] + 0.3 * X[:, 2] > 0.7
    m = sk.RandomForestClassifier(20, min_samples_leaf=2, class_weight="balanced", random_state=0).fit(X, y)
    f = Forest.from_sklearn(m)
    for x, p in zip(X[:30], m.predict_proba(X[:30])[:, 1]):
        assert f.predict(list(x)) == pytest.approx(p, abs=2e-3)


def test_model_saved_and_loaded(tmp_path):
    m = const_model(0.3, 0.7)
    m.save(tmp_path / "m.json")
    m2 = CutModel.load(tmp_path / "m.json")
    assert m2.proba("fast", {}) == 0.3 and m2.proba("full", {}) == 0.7


def test_make_scenecut_choice(tmp_path):
    const_model(0.3, 0.7).save(tmp_path / "m.json")
    assert isinstance(make_scenecut({"people": {"model": str(tmp_path / "m.json")}}, lambda: 0.0), PeopleCutDetector)
    assert isinstance(make_scenecut({"people": {"model": str(tmp_path / "none.json")}}, lambda: 0.0), SceneCutDetector)
    assert isinstance(make_scenecut({"kind": "pixel"}, lambda: 0.0), SceneCutDetector)


def test_shipped_model_loads():
    from pathlib import Path

    m = CutModel.load(Path(__file__).resolve().parent.parent / "config" / "scenecut_model.json")
    assert set(m.stages) == {"fast", "full"} and m.info["check"]["precision"] > 0.7
    assert 0.0 <= m.proba("fast", {k: 0.5 for k in m.features}) <= 1.0


def test_listener_maps_actions_to_controller():
    from soundmaster.listener import Listener
    from soundmaster.media.peoplecut import PeopleCutEvent

    calls = []

    class C:
        state = type("S", (), {"scene_id": 1})()

        def scene_cut(self, *a): calls.append(("cut", a[0]))
        def scene_cut_dip(self, *a): calls.append(("dip", a[0]))
        def scene_cut_reject(self, *a): calls.append(("reject", a[0]))

    class J:
        def write(self, *a, **k): pass

    lis = Listener(C(), J(), None, None, synchronous=True)
    lis._on_people_cut([PeopleCutEvent(a, 1.0, 0.0, 1.0, 0.5, "fast") for a in
                        ("dip", "reject", "confirm", "cut", "late_cut", "doubt", "none")])
    assert calls == [("dip", "people:dip"), ("reject", "people:reject"), ("cut", "people:confirm"),
                     ("cut", "people:cut"), ("cut", "people:late_cut")]


def test_training_features_are_the_detectors_features():
    """Обучение прогоняет сохранённые рамки через тот же поиск проходов, что и детектор на шоу."""
    from soundmaster.learn.model import live_features

    people = scene(True)
    frames = [(k / 5, {i: b for i, b in people(None, k / 5).items() if b[3] - b[1] > 0.12}) for k in range(80)]
    fast, full = live_features(frames, False, 30.0, (5.0, 6.2))
    det = PeopleCutDetector(const_model(0.55, 0.9), people, PeopleCutConfig(), scene_start=lambda: -25.0 + 0)
    evs = run(det)
    assert evs[0].features == pytest.approx({**fast, "scene_age": evs[0].features["scene_age"]})
    assert evs[1].features == pytest.approx({**full, "scene_age": evs[1].features["scene_age"]})
    assert live_features(frames[:20], False, 30.0, (5.0, 6.2)) is None  # прохода в этих кадрах нет


def test_report_reads_model_scene_cut_records(tmp_path):
    """Регрессия: отчёт падал на записях модели «монтажа» (в них нет direction/confidence)."""
    from soundmaster.report import summarize, write_html

    recs = [{"kind": "scene_cut_detected", "t": 5.0, "detector": "people", "action": "dip", "stage": "fast", "p": 0.55},
            {"kind": "scene_cut_detected", "t": 8.0, "detector": "people", "action": "confirm", "stage": "full",
             "p": 0.7, "error": None},
            {"kind": "scene_cut_detected", "t": 20.0, "detector": "pixel", "action": "cut", "stage": "pixel", "p": 0.8,
             "error": "ImportError: No module named 'ultralytics'"},
            {"kind": "scene_cut_detected", "t": 30.0, "action": "none", "stage": "fast", "p": 0.1},
            {"kind": "scene_cut_detected", "t": 40.0, "direction": "left_to_right", "confidence": 0.9}]
    write_html(recs, tmp_path / "r.html", "x")
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "full: confirm" in html and "ultralytics" in html and "fast: none" not in html
    assert summarize(recs)["scene_cuts"] == 3
