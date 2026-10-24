"""Сбор примеров «монтажа»: кандидаты по траекториям, ролики и рамки, разметка."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from soundmaster.learn.passes import PassCandidate, PassFinder, merge


def _walk(finder, x0, x1, seconds, height=0.6, tid=1, fps=5.0, t0=0.0):
    out = []
    n = int(seconds * fps)
    for k in range(n + 1):
        x = x0 + (x1 - x0) * k / n
        out += finder.push(t0 + k / fps, {tid: (x - 0.05, 0.3, x + 0.05, 0.3 + height)})
    return out


def test_fast_big_pass_is_candidate():
    got = _walk(PassFinder(), 0.05, 0.95, 2.0)
    assert len(got) == 1  # один проход — один кандидат, а не на каждом кадре
    c = got[0]
    assert c.track == 1 and c.travel >= 0.3 and c.height == pytest.approx(0.6, abs=0.01)
    assert c.t_end - c.t_start <= 2.0


def test_small_far_or_slow_or_on_spot_is_not_candidate():
    assert not _walk(PassFinder(), 0.05, 0.95, 2.0, height=0.1)  # маленькая фигура в глубине сцены
    assert not _walk(PassFinder(), 0.05, 0.95, 20.0)  # медленная ходьба по сцене
    f = PassFinder()
    got = []
    for k in range(40):  # топчется на месте
        x = 0.5 + 0.08 * np.sin(k)
        got += f.push(k / 5, {1: (x - 0.05, 0.2, x + 0.05, 0.9)})
    assert not got


def test_fast_tracker_keeps_runner_and_standing_people_apart():
    from soundmaster.learn.passes import FastTracker

    tr, ids = FastTracker(), []
    for k in range(8):  # бегун: 17% ширины за кадр (5 кадров/с), рядом стоят двое
        x = 0.05 + 0.17 * k
        got = tr.update(k * 0.2, [(0.30, 0.2, 0.42, 0.85), (x - 0.08, 0.2, x + 0.08, 0.98), (0.62, 0.15, 0.75, 0.9)])
        ids.append(got)
    runner = {next(i for i, b in g.items() if abs((b[0] + b[2]) / 2 - (0.05 + 0.17 * k)) < 0.01)
              for k, g in enumerate(ids)}
    standing = {next(i for i, b in g.items() if b[0] == 0.30) for g in ids}
    assert len(runner) == 1 and len(standing) == 1 and runner != standing


def test_merge_joins_close_moments_and_keeps_people_track():
    a = PassCandidate(10.0, 12.0, -1, 0.7, 0.3, ["pixel"])
    b = PassCandidate(11.0, 12.5, 7, 0.6, 0.5)
    c = PassCandidate(30.0, 31.0, 3, 0.5, 0.4)
    m = merge([c, a, b])
    assert len(m) == 2
    assert m[0].track == 7 and m[0].sources == ["people", "pixel"] and m[0].t_start == 10.0
    assert m[1].track == 3


def _moving_box_video(path, seconds=12, cross_at=4.0):
    """Тёмная сцена, светлая «фигура» в рост проходит слева направо за 1,5 с, начиная с cross_at."""
    x = f"if(lt(t,{cross_at}),-100,(t-{cross_at})*420-100)"
    graph = (f"color=c=0x202020:s=640x360:d={seconds}:r=25[bg];color=c=white:s=90x280:d={seconds}:r=25[fig];"
             f"[bg][fig]overlay=x='{x}':y=60:eval=frame[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-filter_complex", graph, "-map", "[v]", "-pix_fmt", "yuv420p",
                    str(path)], check=True)


def test_process_video_writes_clip_and_boxes(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("нет ffmpeg")
    from soundmaster.learn.collect import H, W, process

    video = tmp_path / "show.mp4"
    _moving_box_video(video)

    def fake_people(rgb, t):  # «нейросеть»: светлое пятно = человек с id 1
        cols = np.nonzero((rgb[..., 0] > 128).any(axis=0))[0]
        rows = np.nonzero((rgb[..., 0] > 128).any(axis=1))[0]
        if len(cols) < 5:
            return {}
        return {1: (cols[0] / W, rows[0] / H, cols[-1] / W, rows[-1] / H)}

    r = process(str(video), "vid1", "Компот — тест", tmp_path / "data", tmp_path / "clips", fake_people,
                progress=lambda s: None)
    assert len(r["candidates"]) == 1
    c = r["candidates"][0]
    assert 3.0 <= c["t_start"] <= 5.5 and "people" in c["sources"] and c["track"] == 1
    assert (tmp_path / "clips" / c["clip"]).stat().st_size > 1000
    assert c["frames"] and all("1" in bx or not bx for _, bx in c["frames"])
    saved = json.loads((tmp_path / "data" / "vid1.json").read_text(encoding="utf-8"))
    assert saved["candidates"][0]["id"] == c["id"] and saved["seconds"] > 10


def test_collect_skips_done_and_orders_shows_first(tmp_path, monkeypatch):
    from soundmaster.learn import collect as col

    entries = [{"id": "a", "title": "Занятие 20.10", "duration": 100, "url": "u1"},
               {"id": "b", "title": "Компот – «9»", "duration": 100, "url": "u2"},
               {"id": "c", "title": "Занятие 13.10", "duration": 100, "url": "u3"}]
    assert [e["id"] for e in col.ordered(entries)] == ["b", "a", "c"]
    monkeypatch.setattr(col, "playlist", lambda url: entries)
    monkeypatch.setattr(col, "ffmpeg_bin", lambda name="ffmpeg": "ffmpeg")
    seen = []

    def fake_process(src, vid, title, out_dir, clip_dir, det, **kw):
        seen.append(vid)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{vid}.json").write_text("{}", encoding="utf-8")
        return {"seconds": 100, "candidates": []}

    monkeypatch.setattr(col, "process", fake_process)
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")  # уже собрано раньше
    col.collect(["https://youtube.com/playlist?list=x"], tmp_path, tmp_path / "clips",
                progress=lambda s: None, detector_factory=lambda: None)
    assert seen == ["b", "c"]


def test_labels_saved_and_listed(tmp_path):
    from soundmaster.learn.label import load_items, load_labels, save_label

    (tmp_path / "v.json").write_text(json.dumps({"video": "v", "title": "Шоу", "candidates": [
        {"id": "v_000040", "clip": "v_000040.mp4", "t_start": 5.5, "sources": ["people"]}]}), encoding="utf-8")
    assert [i["id"] for i in load_items(tmp_path)] == ["v_000040"]
    save_label(tmp_path, "v_000040", "cut")
    assert load_labels(tmp_path) == {"v_000040": "cut"}
    assert [i["id"] for i in load_items(tmp_path)] == ["v_000040"]  # labels.json не считается видео
    save_label(tmp_path, "v_000040", None)
    assert load_labels(tmp_path) == {}
    with pytest.raises(ValueError):
        save_label(tmp_path, "v_000040", "maybe")
