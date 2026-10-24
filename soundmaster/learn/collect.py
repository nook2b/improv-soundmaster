"""Сбор примеров «монтажа» с записей (YouTube-плейлист, ссылка или файл) — без сохранения видео.

Видео идёт с YouTube прямо в анализ (yt-dlp → ffmpeg → кадры в памяти) и на диск не пишется.
По кадрам нейросеть находит людей (YOLO) и ведёт их траектории; там, где кто-то крупный быстро
прошёл поперёк кадра (или сработал старый детектор по пикселям с мягкими порогами), сохраняются:
  - короткий ролик (несколько секунд, без звука) — для разметки, в recordings/ (в git не попадает);
  - рамки людей вокруг этого момента — в data/scenecut/<id видео>.json (числа, без картинок; в git).
На этих рамках и разметке потом учится модель «монтаж / не монтаж».

    python -m soundmaster collect-scenecuts "https://youtube.com/playlist?list=..."

Нужно: yt-dlp (+ deno для YouTube), ffmpeg, `pip install -e ".[learn]"` (ultralytics).
Прерванный сбор можно продолжить той же командой: готовые видео пропускаются.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..media.download import is_url
from ..media.ffmpeg import ffmpeg_bin
from ..media.scenecut import SceneCutConfig, SceneCutDetector
from .passes import Box, FastTracker, PassCandidate, PassFinder, is_scenecut_candidate, merge, pass_end, pass_features

W, H = 640, 360  # размер кадров для анализа и роликов
VERSION = 2  # версия отбора: данные старой версии собираются заново (разметка переносится по времени)
BEFORE, AFTER = 3.0, 7.0  # сколько секунд до и после прохода смотреть, кто стоит на сцене
MAX_CLIP = 6.0  # ролик для разметки: ~5 с вокруг прохода (Иван: 8 с — много)
# Старый детектор по пикселям — с мягкими порогами, только чтобы не пропустить кандидатов.
LOOSE_PIXEL = SceneCutConfig(min_height=0.2, min_travel=0.35, max_duration=6.0, min_monotonic=0.7, cooldown=2.0)


def ytdlp_bin() -> str:
    path = shutil.which("yt-dlp")
    if not path:
        raise RuntimeError("нужен yt-dlp (Windows: winget install yt-dlp.yt-dlp; Mac: brew install yt-dlp)")
    return path


def playlist(url: str) -> list[dict]:
    """Видео плейлиста (или одно видео): [{id, title, duration, url}]."""
    out = subprocess.run([ytdlp_bin(), "--flat-playlist", "-J", url], capture_output=True, text=True,
                         encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"yt-dlp не прочитал {url}: {out.stderr.strip()[-300:]}")
    data = json.loads(out.stdout)
    entries = data.get("entries") or [data]
    return [{"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration") or 0,
             "url": e.get("url") or e.get("webpage_url") or f"https://www.youtube.com/watch?v={e['id']}"}
            for e in entries if e.get("id")]


def is_show(title: str) -> bool:
    t = title.lower()
    return "компот" in t or "шоу" in t


def ordered(entries: list[dict]) -> list[dict]:
    """Сначала шоу (так будет на площадке), потом занятия — в порядке плейлиста (новые первыми)."""
    return sorted(entries, key=lambda e: not is_show(e["title"]))


def frames(source: str, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Кадры W×H RGB с частотой fps: из ссылки (поток, без файла на диске) или из файла."""
    vf = (f"fps={fps},scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2")
    ff = [ffmpeg_bin(), "-v", "error", "-i", "pipe:0" if is_url(source) else source, "-an",
          "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    procs = []
    if is_url(source):
        # только видео (без звука), не выше 480p: для анализа хватает, трафика меньше
        yt = subprocess.Popen([ytdlp_bin(), "-q", "--no-warnings", "--no-playlist", "-f",
                               "bv*[height<=480][vcodec^=avc1]/bv*[height<=480]/b[height<=480]/b",
                               "-o", "-", source], stdout=subprocess.PIPE)
        procs.append(yt)
        ffp = subprocess.Popen(ff, stdin=yt.stdout, stdout=subprocess.PIPE)
        assert yt.stdout is not None
        yt.stdout.close()  # чтобы yt-dlp узнал, если ffmpeg закончит раньше
    else:
        ffp = subprocess.Popen(ff, stdout=subprocess.PIPE)
    procs.append(ffp)
    assert ffp.stdout is not None
    size = W * H * 3
    i = 0
    try:
        while True:
            raw = ffp.stdout.read(size)
            if len(raw) < size:
                break
            yield i / fps, np.frombuffer(raw, np.uint8).reshape(H, W, 3)
            i += 1
    finally:
        for p in procs:
            p.kill()
            p.wait()


_MODELS: dict[str, object] = {}


class YoloPeople:
    """Люди в кадре с id траекторий: YOLO находит рамки, FastTracker связывает их между кадрами.

    Один объект — одно видео (траектории не переходят между записями); нейросеть грузится один раз.
    """

    # yolo11s, а не n: на проходе перед камерой человек смазан и обрезан краем кадра — «s» находит его
    # надёжнее, а медленнее всего в ~1,6 раза (замер 2026-10-24: 36 против 23 мс на кадр на CPU).
    def __init__(self, model: str = "yolo11s.pt", imgsz: int = 416, conf: float = 0.25):
        if model not in _MODELS:
            from ultralytics import YOLO  # type: ignore

            _MODELS[model] = YOLO(model)
        self.model = _MODELS[model]
        self.imgsz, self.conf = imgsz, conf
        self.tracker = FastTracker()
        self.name = Path(model).stem

    def __call__(self, rgb: np.ndarray, t: float) -> dict[int, Box]:
        r = self.model.predict(rgb[..., ::-1], classes=[0], imgsz=self.imgsz, conf=self.conf,  # type: ignore[attr-defined]
                               verbose=False)[0]
        dets = [tuple(float(v) for v in b) for b in r.boxes.xyxyn.cpu().numpy()] if r.boxes is not None else []
        return self.tracker.update(t, dets)  # type: ignore[arg-type]


def draw_box(rgb: np.ndarray, box: Box, color=(255, 220, 0), t: int = 2) -> np.ndarray:
    out = rgb.copy()
    x1, y1, x2, y2 = (int(round(v * s)) for v, s in zip(box, (W, H, W, H)))
    x1, x2 = max(0, min(W - 1, x1)), max(0, min(W - 1, x2))
    y1, y2 = max(0, min(H - 1, y1)), max(0, min(H - 1, y2))
    out[y1:y1 + t, x1:x2 + 1] = color
    out[max(0, y2 - t + 1):y2 + 1, x1:x2 + 1] = color
    out[y1:y2 + 1, x1:x1 + t] = color
    out[y1:y2 + 1, max(0, x2 - t + 1):x2 + 1] = color
    return out


def write_clip(path: Path, imgs: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.mp4")
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                    "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
                   input=b"".join(np.ascontiguousarray(f).tobytes() for f in imgs), check=True,
                   capture_output=True)
    tmp.replace(path)


def process(source: str, vid: str, title: str, out_dir: Path, clip_dir: Path, detector: Callable,
            fps: float = 10.0, detect_every: int = 2, progress: Callable[[str], None] = print,
            duration: float = 0.0) -> dict:
    """Пройти одно видео: кандидаты → ролики + рамки людей. Возвращает и пишет <out_dir>/<vid>.json."""
    ring: deque[tuple[float, np.ndarray]] = deque(maxlen=int(fps * (MAX_CLIP + AFTER + 4)))
    log: deque[tuple[float, dict[int, Box]]] = deque(
        maxlen=int(fps / detect_every * (BEFORE + AFTER + MAX_CLIP + 10)))
    finder, pixel = PassFinder(), SceneCutDetector(LOOSE_PIXEL)
    pending: list[PassCandidate] = []
    done: list[dict] = []
    skipped = 0
    t0, t, last_report = time.monotonic(), 0.0, 0.0

    def finish(c: PassCandidate) -> None:
        nonlocal skipped
        window = [(lt, bx) for lt, bx in log if c.t_start - BEFORE <= lt <= c.t_end + AFTER]
        feats = pass_features(c.track, c.sources, window, c.t_start, c.t_end, BEFORE, AFTER)
        if c.track < 0 or not is_scenecut_candidate(feats):
            skipped += 1
            return
        end = pass_end(c.track, window, c.t_end)
        a = max(c.t_start - 1.5, c.t_end - 3.5)  # t_end — момент, когда фигура прошла заметную часть кадра
        b = min(max(end + 1.5, a + 5.0), a + MAX_CLIP)  # ~5 с: проход целиком и немного после
        cid = f"{vid}_{int(c.t_start * 10):06d}"
        imgs = []
        for ft, img in ring:
            if a <= ft <= b:
                near = min(window, key=lambda p: abs(p[0] - ft)) if window else None
                if near and abs(near[0] - ft) < 0.5 and c.track in near[1]:
                    img = draw_box(img, near[1][c.track])
                imgs.append(img)
        if len(imgs) < fps:  # начало видео — ролик слишком короткий
            return
        write_clip(clip_dir / f"{cid}.mp4", imgs, fps)
        done.append({"id": cid, "clip": f"{cid}.mp4", "t_clip": round(a, 2), "t_pass_end": round(end, 2),
                     **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(c).items()},
                     "features": {k: round(float(v), 3) for k, v in feats.items()},
                     "frames": [[round(lt, 2), {str(k): [round(v, 3) for v in bx] for k, bx in bxs.items()
                                                if bx[3] - bx[1] > 0.12}]  # мелкие фигуры в глубине не нужны
                                for lt, bxs in window]})

    for i, (t, rgb) in enumerate(frames(source, fps)):
        ring.append((t, rgb))
        ev = pixel.push(t, rgb)
        if ev:
            pending.append(PassCandidate(ev.t_start, ev.t, -1, ev.travel, ev.height, ["pixel"]))
        if i % detect_every == 0:
            boxes = detector(rgb, t)
            log.append((t, boxes))
            pending += finder.push(t, boxes)
        if pending:
            pending = merge(pending)
            for c in [c for c in pending if t >= c.t_end + AFTER]:
                pending.remove(c)
                finish(c)
        if t - last_report >= 300:
            last_report = t
            speed = t / max(1e-6, time.monotonic() - t0)
            left = f", осталось ~{(duration - t) / speed / 60:.0f} мин" if duration else ""
            progress(f"  {title[:40]}: {t / 60:.0f} мин, кандидатов {len(done)} (отсеяно {skipped}), "
                     f"скорость ×{speed:.1f}{left}")
    for c in merge(pending):
        finish(c)
    result = {"video": vid, "title": title, "version": VERSION, "detector": getattr(detector, "name", ""),
              "source": source if is_url(source) else Path(source).name, "seconds": round(t, 1), "fps": fps,
              "detect_fps": fps / detect_every, "size": [W, H], "made": time.strftime("%Y-%m-%d %H:%M"),
              "skipped": skipped, "candidates": done}
    out_dir.mkdir(parents=True, exist_ok=True)
    carry_labels(out_dir, vid, done, progress)
    tmp = out_dir / f"{vid}.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(out_dir / f"{vid}.json")
    return result


def carry_labels(out_dir: Path, vid: str, new: list[dict], progress: Callable[[str], None] = print) -> None:
    """Перенести разметку прошлого сбора этого видео на новых кандидатов (по пересечению по времени)."""
    from .label import load_labels, save_labels

    old_path = out_dir / f"{vid}.json"
    labels = load_labels(out_dir)
    if not old_path.exists():
        return
    old = json.loads(old_path.read_text(encoding="utf-8")).get("candidates", [])
    old_lab = [(o, labels.pop(o["id"])) for o in old if o["id"] in labels]
    if not old_lab:
        return
    moved = lost_cuts = 0
    for o, lab in old_lab:
        hit = next((n for n in new if n["t_start"] - 1.0 <= o["t_end"] and o["t_start"] <= n["t_pass_end"] + 1.0), None)
        if hit and (hit["id"] not in labels or lab == "cut"):
            labels[hit["id"]] = lab
            moved += 1
        elif lab == "cut" and not hit:
            lost_cuts += 1
            progress(f"  монтаж на {int(o['t_start'] // 60)}:{int(o['t_start'] % 60):02d} не прошёл новый отбор")
    save_labels(out_dir, labels)
    progress(f"  разметка перенесена: {moved} из {len(old_lab)}" + (f", монтажей потеряно: {lost_cuts}" if lost_cuts else ""))


def collect(sources: list[str], out_dir: Path, clip_dir: Path, limit: int | None = None, only_shows: bool = False,
            fps: float = 10.0, progress: Callable[[str], None] = print,
            detector_factory: Callable[[], Callable] | None = None, model: str = "yolo11s.pt") -> list[dict]:
    """Собрать примеры со всех источников (плейлисты/ссылки/файлы); готовые видео пропускаются."""
    ffmpeg_bin()
    make = detector_factory or (lambda: YoloPeople(model))
    items: list[dict] = []
    for s in sources:
        if is_url(s):
            items += playlist(s)
        else:
            p = Path(s)
            items.append({"id": p.stem, "title": p.name, "duration": 0, "url": str(p)})
    items = ordered(items)
    if only_shows:
        items = [e for e in items if is_show(e["title"])]
    def ready(e: dict) -> bool:
        p = out_dir / f"{e['id']}.json"
        if not p.exists():
            return False
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("version", 1) >= VERSION
        except ValueError:
            return False

    todo = [e for e in items if not ready(e)]
    ready = len(items) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    progress(f"Видео: {len(items)}, уже собрано: {ready}, сейчас: {len(todo)} "
             f"(~{sum(e['duration'] for e in todo) / 3600:.0f} ч записи)")
    results = []
    for n, e in enumerate(todo, 1):
        progress(f"[{n}/{len(todo)}] {e['title']}")
        try:
            r = process(e["url"], e["id"], e["title"], out_dir, clip_dir, make(), fps=fps, progress=progress,
                        duration=e["duration"])
        except Exception as ex:  # noqa: BLE001 — одно видео не должно останавливать остальные
            progress(f"  пропущено: {type(ex).__name__}: {str(ex)[:300]}")
            continue
        if r["seconds"] < 0.5 * e["duration"]:
            progress(f"  поток оборвался на {r['seconds'] / 60:.0f} мин из {e['duration'] / 60:.0f} — "
                     "видео будет пройдено заново при следующем запуске")
            (out_dir / f"{e['id']}.json").unlink()
            continue
        progress(f"  готово: кандидатов {len(r['candidates'])}")
        results.append(r)
    return results
