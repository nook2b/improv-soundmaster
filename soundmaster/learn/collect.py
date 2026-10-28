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
from .signals import AUDIO_SR, AudioTrack, pack, pack_pose, pose_points
from ..media.ffmpeg import ffmpeg_bin
from ..media.scenecut import SceneCutConfig, SceneCutDetector
from .passes import Box, FastTracker, PassCandidate, PassFinder, is_scenecut_candidate, merge, pass_end, pass_features

W, H = 640, 360  # размер кадров для анализа и роликов
VERSION = 4  # версия отбора/данных: старые собираются заново (разметка переносится по времени); 4 — поза, звук, кадр
BEFORE, AFTER = 3.0, 7.0  # сколько секунд до и после прохода смотреть, кто стоит на сцене
MAX_SPAN = 12.0  # кандидат (с учётом слияния соседних) не длиннее — иначе не помещается в журнал рамок
MAX_CLIP = 7.0  # ролик для разметки: ~5–7 с, с момента, как человек появился в кадре (Иван: 8 с — много)
EMB_EVERY = 0.5  # описание кадра: из кадров раз в 0,5 с (считается только для кандидатов)
WIDE_BEFORE = 5.0  # «Шире» на странице разметки: столько секунд до прохода и AFTER после —
# видно, вышел ли человек из-за кулис и ушли ли потом актёры
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


def audio_cmd(source: str) -> list[list[str]]:
    """Команды звукового потока (моно 16 кГц, s16le): из ссылки — отдельно только звук (yt-dlp), из файла — ffmpeg."""
    ff = [ffmpeg_bin(), "-v", "error", "-i", "pipe:0" if is_url(source) else source, "-vn", "-ac", "1",
          "-ar", str(AUDIO_SR), "-f", "s16le", "-"]
    if is_url(source):
        return [[ytdlp_bin(), "-q", "--no-warnings", "--no-playlist", "-f", "ba/b", "-o", "-", source], ff]
    return [ff]


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
        # конец потока: убедиться, что это конец видео, а не ошибка YouTube/ffmpeg
        codes = [p.wait(timeout=60) for p in reversed(procs)]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"поток оборвался на {i / fps / 60:.0f} мин (коды завершения {codes})")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.wait()


_MODELS: dict[str, object] = {}


class YoloPeople:
    """Люди в кадре с id траекторий: YOLO находит рамки, FastTracker связывает их между кадрами.

    Один объект — одно видео (траектории не переходят между записями); нейросеть грузится один раз.
    """

    # yolo11s, а не n: на проходе перед камерой человек смазан и обрезан краем кадра — «s» находит его
    # надёжнее, а медленнее всего в ~1,6 раза (замер 2026-10-24: 36 против 23 мс на кадр на CPU).
    # -pose: те же рамки людей плюс точки тела (решение Ивана 2026-10-28), скорость та же (59 против 58 мс).
    def __init__(self, model: str = "yolo11s-pose.pt", imgsz: int = 416, conf: float = 0.25):
        if model not in _MODELS:
            from ultralytics import YOLO  # type: ignore

            _MODELS[model] = YOLO(model)
        self.model = _MODELS[model]
        self.imgsz, self.conf = imgsz, conf
        self.tracker = FastTracker()
        self.name = Path(model).stem
        self.last_pose: dict[int, list[float]] = {}  # точки тела людей последнего кадра (если модель -pose)

    def __call__(self, rgb: np.ndarray, t: float) -> dict[int, Box]:
        r = self.model.predict(rgb[..., ::-1], classes=[0], imgsz=self.imgsz, conf=self.conf,  # type: ignore[attr-defined]
                               verbose=False)[0]
        dets = [tuple(float(v) for v in b) for b in r.boxes.xyxyn.cpu().numpy()] if r.boxes is not None else []
        kps: list[list[float] | None] = [None] * len(dets)
        if getattr(r, "keypoints", None) is not None and len(r.keypoints.xyn) == len(dets):
            conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            xy = r.keypoints.xyn.cpu().numpy()
            kps = [pose_points(xy[i], conf[i] if conf is not None else None) for i in range(len(dets))]
        boxes = self.tracker.update(t, dets)  # type: ignore[arg-type]
        by_box = {d: k for d, k in zip(dets, kps) if k is not None}
        self.last_pose = {tid: by_box[b] for tid, b in boxes.items() if b in by_box}
        return boxes


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


def manual_candidate(tm: float, log: list[tuple[float, dict[int, Box]]]) -> PassCandidate:
    """Монтаж, который Иван отметил вручную (время в записи): берём крупную фигуру, прошедшую дальше всех рядом."""
    tracks: dict[int, list[tuple[float, float]]] = {}
    for lt, bx in log:
        if tm - 4.0 <= lt <= tm + 4.0:
            for k, b in bx.items():
                if b[3] - b[1] >= 0.3:
                    tracks.setdefault(k, []).append((lt, (b[0] + b[2]) / 2))
    best = max(((max(x for _, x in p) - min(x for _, x in p), k, p) for k, p in tracks.items() if len(p) >= 3),
               default=None)
    if best is None:
        return PassCandidate(tm - 1.0, tm + 1.0, -1, 0.0, 0.0, ["manual"])
    span, k, pts = best
    return PassCandidate(pts[0][0], pts[-1][0], k, span, 0.0, ["manual"])


def process(source: str, vid: str, title: str, out_dir: Path, clip_dir: Path, detector: Callable,
            fps: float = 10.0, detect_every: int = 2, progress: Callable[[str], None] = print,
            duration: float = 0.0, manual: list[float] | None = None, save_clips: bool = False,
            embedder: Callable[[np.ndarray], np.ndarray] | None = None, audio: bool = True) -> dict:
    """Пройти одно видео: кандидаты → ролики + рамки людей. Возвращает и пишет <out_dir>/<vid>.json.

    manual — моменты (секунды записи), которые Иван отметил как пропущенный монтаж: по ним кандидат
    создаётся всегда, без отбора, и сразу получает метку «монтаж».
    save_clips — нарезать ролики mp4 (страница разметки их больше не использует: видео играет с YouTube,
    рамки рисуются поверх; ролики нужны только для записей не с YouTube).
    embedder — описание кадра (signals.FrameEmbedder): считается только для кандидатов, в EMB_AT моментах.
    audio — признаки звука (signals.AudioTrack) вокруг кандидата; нет звука — без них.
    """
    manual_q = sorted(manual or [])
    # кандидат не длиннее MAX_SPAN (merge), плюс запас до/после: журнал рамок и кадры должны его вмещать
    keep_s = WIDE_BEFORE + MAX_SPAN + AFTER + 6
    ring: deque[tuple[float, np.ndarray]] = deque(maxlen=int(fps * keep_s) if save_clips else 1)
    log: deque[tuple[float, dict[int, Box]]] = deque(maxlen=int(fps / detect_every * keep_s))
    pose_log: deque[tuple[float, dict[int, list[float]]]] = deque(maxlen=log.maxlen)
    emb_ring: deque[tuple[float, np.ndarray]] = deque(maxlen=int(keep_s / EMB_EVERY) + 2)  # кадры 2 раза в с
    emb_cache: dict[float, np.ndarray] = {}
    track_audio = AudioTrack(audio_cmd(source) if audio else None)
    finder, pixel = PassFinder(), SceneCutDetector(LOOSE_PIXEL)
    pending: list[PassCandidate] = []
    done: list[dict] = []
    skipped = 0
    t0, t, last_report = time.monotonic(), 0.0, 0.0

    def finish(c: PassCandidate) -> None:
        nonlocal skipped
        window = [(lt, bx) for lt, bx in log if c.t_start - BEFORE <= lt <= c.t_end + AFTER]
        feats = pass_features(c.track, c.sources, window, c.t_start, c.t_end, BEFORE, AFTER)
        forced = "manual" in c.sources
        if not forced and (c.track < 0 or not is_scenecut_candidate(feats)):
            skipped += 1
            return
        end = pass_end(c.track, window, c.t_end)
        seen = first_seen(c.track, list(log), c.t_start)
        a = min(seen, c.t_start) - 1.0  # с момента, как человек появился (вышел из-за кулис)
        b = min(max(end + 1.5, a + 5.0), a + MAX_CLIP)
        n = int(c.t_start * 10)
        ids = {d["id"] for d in done}
        while f"{vid}_{n:06d}" in ids:  # два кандидата с одним началом — разные id
            n += 1
        cid = f"{vid}_{n:06d}"
        boxes = list(log)

        def shots(a: float, b: float) -> list[np.ndarray]:
            imgs = []
            for ft, img in ring:
                if a <= ft <= b:
                    near = min(boxes, key=lambda p: abs(p[0] - ft)) if boxes else None
                    if near and abs(near[0] - ft) < 0.5 and c.track in near[1]:
                        img = draw_box(img, near[1][c.track])
                    imgs.append(img)
            return imgs

        if a < 0.5 or not any(lt >= b - 0.5 for lt, _ in log):  # самое начало/конец записи — отрезка нет
            if not forced:
                return
        clip = wide_clip = None
        if save_clips:
            imgs = shots(a, b)
            write_clip(clip_dir / f"{cid}.mp4", imgs, fps)
            clip = f"{cid}.mp4"
            wide = shots(c.t_start - WIDE_BEFORE, c.t_end + AFTER)
            if len(wide) > len(imgs):
                write_clip(clip_dir / f"{cid}_wide.mp4", wide, fps)
                wide_clip = f"{cid}_wide.mp4"
        extra: dict = {}
        # точки тела крупных людей (как актёры в pass_features) — от начала прохода до 3 с после; сжато (pack_pose)
        big = {lt: {k for k, b in bx.items() if b[3] - b[1] > 0.3} for lt, bx in log}
        pw = [[round(lt, 2), {str(k): pack_pose(kp) for k, kp in kps.items() if k in big.get(lt, ())}]
              for lt, kps in pose_log if c.t_start - 1.0 <= lt <= end + 3.0]
        if any(p for _, p in pw):
            extra["pose"] = pw
        if embedder is not None and emb_ring:
            moments = [c.t_start - 2.0, c.t_start, c.t_end, end, end + 2.0]
            vecs = []
            for m in moments:
                ft, img = min(emb_ring, key=lambda f: abs(f[0] - m))
                if ft not in emb_cache:
                    emb_cache[ft] = embedder(img)
                vecs.append(emb_cache[ft])
            extra["emb"] = {"t": [round(m, 2) for m in moments], "v": pack(vecs)}
        aud = track_audio.window(c.t_start - 8.0, end + 3.0)
        if aud:
            extra["audio"] = aud
        done.append({"id": cid, "clip": clip, "clip_wide": wide_clip,
                     "t_clip": round(a, 2), "t_seen": round(seen, 2), "t_pass_end": round(end, 2),
                     **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(c).items()},
                     "features": {k: round(float(v), 3) for k, v in feats.items()},
                     "frames": [[round(lt, 2), {str(k): [round(v, 3) for v in bx] for k, bx in bxs.items()
                                                if bx[3] - bx[1] > 0.12}]  # мелкие фигуры в глубине не нужны
                                for lt, bxs in window], **extra})

    def add_manual(tm: float) -> None:
        """Ручная отметка: тот же кандидат, если программа уже нашла проход в этом месте (как кнопка M
        на странице — в пределах 4 с от начала прохода), иначе — новый, через pending (полное окно «после»)."""
        hit = next((d for d in done if abs(d["t_start"] - tm) < 4), None)
        near = [c for c in pending if abs(c.t_start - tm) < 4]
        if hit:
            hit["sources"] = sorted(set(hit["sources"]) | {"manual"})
        elif near:
            for c in near:
                c.sources = sorted(set(c.sources) | {"manual"})
        else:
            pending.append(manual_candidate(tm, list(log)))

    for i, (t, rgb) in enumerate(frames(source, fps)):
        ring.append((t, rgb))
        if embedder is not None and (not emb_ring or t - emb_ring[-1][0] >= EMB_EVERY - 1e-6):
            emb_ring.append((t, rgb))
        ev = pixel.push(t, rgb)
        if ev:
            pending.append(PassCandidate(ev.t_start, ev.t, -1, ev.travel, ev.height, ["pixel"]))
        if i % detect_every == 0:
            boxes = detector(rgb, t)
            log.append((t, boxes))
            pose_log.append((t, dict(getattr(detector, "last_pose", None) or {})))
            pending += finder.push(t, boxes)
        while manual_q and t >= manual_q[0] + 4.0:
            add_manual(manual_q.pop(0))
        if pending:
            pending = merge(pending, max_span=MAX_SPAN)
            for c in [c for c in pending if t >= c.t_end + AFTER]:
                pending.remove(c)
                finish(c)
        if t - last_report >= 300:
            last_report = t
            speed = t / max(1e-6, time.monotonic() - t0)
            left = f", осталось ~{(duration - t) / speed / 60:.0f} мин" if duration else ""
            progress(f"  {title[:40]}: {t / 60:.0f} мин, кандидатов {len(done)} (отсеяно {skipped}), "
                     f"скорость ×{speed:.1f}{left}")
    for tm in manual_q:  # отмечено у самого конца записи
        if tm <= t:
            add_manual(tm)
    for c in merge(pending, max_span=MAX_SPAN):
        finish(c)
    if duration and t < duration - 60:
        # поток кончился раньше видео: не трогаем прошлый сбор и разметку этого видео
        raise RuntimeError(f"поток оборвался на {t / 60:.0f} мин из {duration / 60:.0f}")
    track_audio.close()
    result = {"video": vid, "title": title, "version": VERSION, "detector": getattr(detector, "name", ""),
              "embedder": getattr(embedder, "name", None) if embedder else None,
              "audio": track_audio.error or bool(track_audio.feats),
              "source": source if is_url(source) else Path(source).name, "seconds": round(t, 1), "fps": fps,
              "detect_fps": fps / detect_every, "size": [W, H], "made": time.strftime("%Y-%m-%d %H:%M"),
              "skipped": skipped, "manual": sorted(manual or []), "candidates": done}
    out_dir.mkdir(parents=True, exist_ok=True)
    carry_labels(out_dir, vid, done, progress)
    from .label import load_labels, save_labels

    labels = load_labels(out_dir)
    marked = [c["id"] for c in done if "manual" in c["sources"] and c["id"] not in labels]
    if marked:
        labels.update({cid: "cut" for cid in marked})
        save_labels(out_dir, labels)
    # ролики прошлых сборов этого видео больше не нужны — не копим их на диске
    keep = {c["clip"] for c in done if c.get("clip")} | {c["clip_wide"] for c in done if c.get("clip_wide")}
    for old in (clip_dir.glob(f"{vid}_*.mp4") if clip_dir.exists() else []):
        if old.name not in keep:
            old.unlink(missing_ok=True)
    tmp = out_dir / f"{vid}.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(out_dir / f"{vid}.json")
    return result


def default_embedder(progress: Callable[[str], None] = print):
    """Описание кадра (DINOv2) — если установлен timm; иначе сбор идёт без него (с предупреждением)."""
    from .signals import FrameEmbedder

    try:
        return FrameEmbedder()
    except Exception as e:  # noqa: BLE001
        progress(f"Описание кадра не включено ({type(e).__name__}: {e}) — установите: pip install -e \".[learn]\"")
        return None


def first_seen(track: int, log: list[tuple[float, dict[int, Box]]], t_start: float, limit: float = 4.0) -> float:
    """Когда этот человек появился в кадре: идём назад от начала прохода, пока он виден (не дальше limit с)."""
    seen = t_start
    for lt, bx in reversed(log):
        if lt > t_start:
            continue
        if track not in bx or t_start - lt > limit:
            break
        seen = lt
    return seen


def same_person(old: dict, track: str, new: dict) -> str | None:
    """id траектории в новом кандидате, рамка которой больше всего совпадает с указанной в старом."""
    def iou(a: list[float], b: list[float]) -> float:
        w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = w * h
        return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    score: dict[str, float] = {}
    nf = new.get("frames", [])
    for t, bx in old.get("frames", []):
        if track not in bx or not nf:
            continue
        nt, nbx = min(nf, key=lambda f: abs(f[0] - t))
        if abs(nt - t) > 0.3:
            continue
        for k, b in nbx.items():
            score[k] = score.get(k, 0.0) + iou(bx[track], b)
    return max(score, key=score.get) if score and max(score.values()) > 0 else None


def carry_labels(out_dir: Path, vid: str, new: list[dict], progress: Callable[[str], None] = print) -> None:
    """Перенести разметку прошлого сбора этого видео на новых кандидатов (по пересечению по времени)."""
    from .label import load_labels, load_tracks, save_labels, save_track

    old_path = out_dir / f"{vid}.json"
    labels = load_labels(out_dir)
    if not old_path.exists():
        return
    old = json.loads(old_path.read_text(encoding="utf-8")).get("candidates", [])
    fixed = load_tracks(out_dir)
    old_lab = [(o, labels.pop(o["id"])) for o in old if o["id"] in labels]
    if not old_lab:
        return
    moved = lost_cuts = 0
    for o, lab in old_lab:
        hit = next((n for n in new if n["t_start"] - 1.0 <= o["t_end"] and o["t_start"] <= n["t_pass_end"] + 1.0), None)
        is_cut = lab in ("cut", "cut_other")
        if hit and (hit["id"] not in labels or is_cut):
            labels[hit["id"]] = lab
            moved += 1
            if o["id"] in fixed:  # человек, указанный вручную: та же рамка в новом сборе (id траекторий меняются)
                save_track(out_dir, o["id"], None)
                tr = same_person(o, fixed[o["id"]], hit)
                if tr is not None:
                    save_track(out_dir, hit["id"], tr)
        elif o["id"] in fixed:
            save_track(out_dir, o["id"], None)
        if is_cut and not hit:
            lost_cuts += 1
            progress(f"  монтаж на {int(o['t_start'] // 60)}:{int(o['t_start'] % 60):02d} не прошёл новый отбор")
    save_labels(out_dir, labels)
    progress(f"  разметка перенесена: {moved} из {len(old_lab)}" + (f", монтажей потеряно: {lost_cuts}" if lost_cuts else ""))


def collect(sources: list[str], out_dir: Path, clip_dir: Path, limit: int | None = None, only_shows: bool = False,
            fps: float = 10.0, progress: Callable[[str], None] = print,
            detector_factory: Callable[[], Callable] | None = None, model: str = "yolo11s-pose.pt",
            save_clips: bool = False, push: bool = False, labeled_only: bool = False,
            embedder_factory: Callable[[], Callable | None] | None = None) -> list[dict]:
    """Собрать примеры со всех источников (плейлисты/ссылки/файлы); готовые видео пропускаются.

    labeled_only — только видео, где уже есть разметка (пересбор для новых признаков: размечать заново не нужно).
    """
    ffmpeg_bin()
    make = detector_factory or (lambda: YoloPeople(model))
    embedder = embedder_factory() if embedder_factory else default_embedder(progress)
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
    if labeled_only:
        from .label import load_labels

        labeled = {cid.rsplit("_", 1)[0] for cid in load_labels(out_dir)}
        items = [e for e in items if e["id"] in labeled]
    from .label import load_manual

    manual = load_manual(out_dir)

    def ready(e: dict) -> bool:
        p = out_dir / f"{e['id']}.json"
        if not p.exists():
            return False
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return False
        # новые ручные отметки пропущенных монтажей — видео собирается заново
        return d.get("version", 1) >= VERSION and set(manual.get(e["id"], [])) <= set(d.get("manual", []))

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
                        manual=manual.get(e["id"], []), save_clips=save_clips,
                        duration=e["duration"], embedder=embedder)
        except Exception as ex:  # noqa: BLE001 — одно видео не должно останавливать остальные
            progress(f"  пропущено: {type(ex).__name__}: {str(ex)[:300]} — прошлый сбор и разметка этого видео "
                     "не тронуты, при следующем запуске видео будет пройдено заново")
            continue
        progress(f"  готово: кандидатов {len(r['candidates'])}")
        results.append(r)
        if push:  # отправить на GitHub сразу, не дожидаясь конца всего плейлиста
            from .sync import GitSync

            res = GitSync(out_dir).push(f"Сбор монтажей: {e['title']}")
            progress(f"  GitHub: {res.message}")
    return results
