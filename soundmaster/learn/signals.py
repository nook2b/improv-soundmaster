"""Дополнительные сигналы для модели «монтажа» (пересбор v4, решение Ивана 2026-10-28): поза, звук, описание кадра.

Всё считается одинаково при сборе примеров (`collect.py`) и на шоу — чтобы модель видела те же числа.
Хранятся только числа, без картинок и звука.

  - поза: YOLO-pose вместо обычной YOLO (та же скорость) — 13 точек тела крупных людей;
  - звук: каждые 0,25 с громкость, энергия 6 полос, «шумность» (аплодисменты — ровный шум) и «яркость»;
  - описание кадра: DINOv2-small (timm, только веса) — 384 числа на кадр; считается не на каждом кадре,
    а на 5 моментах вокруг прохода.
"""

from __future__ import annotations

import base64
import subprocess
import threading
import time
from typing import Iterable

import numpy as np

# нос, плечи, локти, запястья, бёдра, колени, лодыжки (индексы COCO; глаза и уши не нужны)
POSE_POINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
POSE_MIN_CONF = 0.3

AUDIO_SR = 16000
AUDIO_HOP = 0.25
AUDIO_BANDS = (0, 250, 500, 1000, 2000, 4000, 8000)  # Гц
AUDIO_NAMES = ("rms", "b0", "b1", "b2", "b3", "b4", "b5", "flat", "centroid")


def pose_points(kp_xy: np.ndarray, kp_conf: np.ndarray | None) -> list[float]:
    """17 точек COCO (доли кадра) → 26 чисел (x, y для 13 точек), невидимая точка — −1.

    В файлах сбора хранится сжато (pack_pose): 1 байт на координату, шаг 1/250 кадра."""
    out: list[float] = []
    for i in POSE_POINTS:
        if kp_conf is not None and kp_conf[i] < POSE_MIN_CONF:
            out += [-1.0, -1.0]
        else:
            out += [round(float(kp_xy[i][0]), 3), round(float(kp_xy[i][1]), 3)]
    return out


def audio_frame_features(x: np.ndarray, sr: int = AUDIO_SR) -> list[float]:
    """Один отрезок звука (0,25 с, моно float) → 9 чисел (AUDIO_NAMES): дБ, полосы в дБ, плоскость, центроид."""
    x = x.astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    bands = [10 * np.log10(spec[(freqs >= a) & (freqs < b)].sum() + 1e-12)
             for a, b in zip(AUDIO_BANDS, AUDIO_BANDS[1:])]
    sel = spec[(freqs >= 100) & (freqs < 8000)]
    flat = float(np.exp(np.mean(np.log(sel))) / np.mean(sel))  # 1 — ровный шум (аплодисменты), 0 — тон
    centroid = float((freqs * spec).sum() / spec.sum()) / (sr / 2)
    return [round(20 * np.log10(rms), 1), *(round(float(b), 1) for b in bands), round(flat, 3), round(centroid, 3)]


class AudioTrack:
    """Звук записи в своём потоке: признаки каждые AUDIO_HOP с. Без звука (или при ошибке) — пусто, сбор идёт."""

    def __init__(self, cmd: list[list[str]] | None):
        self.feats: list[list[float]] = []
        self.done = threading.Event()
        self.error: str | None = None
        self._procs: list[subprocess.Popen] = []
        if cmd is None:
            self.done.set()
            return
        self._thread = threading.Thread(target=self._run, args=(cmd,), name="collect-audio", daemon=True)
        self._thread.start()

    @property
    def t_done(self) -> float:
        return len(self.feats) * AUDIO_HOP

    def _run(self, cmds: list[list[str]]) -> None:
        try:
            prev = None
            for c in cmds:
                p = subprocess.Popen(c, stdin=prev.stdout if prev else None, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
                if prev is not None and prev.stdout is not None:
                    prev.stdout.close()
                self._procs.append(p)
                prev = p
            assert prev is not None and prev.stdout is not None
            n = int(AUDIO_SR * AUDIO_HOP)
            while True:
                raw = prev.stdout.read(n * 2)
                if len(raw) < n * 2:
                    break
                x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                self.feats.append(audio_frame_features(x))
        except Exception as e:  # noqa: BLE001 — звук не обязателен
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.close()
            self.done.set()

    def close(self) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.kill()
            p.wait()

    def window(self, t0: float, t1: float, wait: float = 120.0) -> dict | None:
        """Признаки на [t0, t1]: {"t0", "hop", "f": [[9 чисел], …]}; ждёт, пока поток звука дойдёт до t1."""
        deadline = time.monotonic() + wait
        while self.t_done < t1 and not self.done.is_set() and time.monotonic() < deadline:
            self.done.wait(timeout=0.2)
        if not self.feats:
            return None
        a, b = max(0, int(t0 / AUDIO_HOP)), min(len(self.feats), int(t1 / AUDIO_HOP) + 1)
        if a >= b:
            return None
        return {"t0": round(a * AUDIO_HOP, 2), "hop": AUDIO_HOP, "f": self.feats[a:b]}


class FrameEmbedder:
    """Описание кадра готовой нейросетью DINOv2-small (timm; скачиваются только веса). Кадр RGB → 384 числа."""

    MODEL = "vit_small_patch14_dinov2.lvd142m"

    def __init__(self, model: str = MODEL, size: int = 224):
        import timm  # type: ignore
        import torch  # type: ignore

        self.torch = torch
        self.net = timm.create_model(model, pretrained=True, num_classes=0, img_size=size).eval()
        self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self.name = model

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        ys = (np.arange(self.size) * h / self.size).astype(int)
        xs = (np.arange(self.size) * w / self.size).astype(int)
        img = (rgb[ys][:, xs].astype(np.float32) / 255.0 - self.mean) / self.std
        x = self.torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]
        with self.torch.no_grad():
            return self.net(x)[0].numpy().astype(np.float32)


def pack_pose(points: list[float]) -> str:
    """26 координат (доли кадра, −1 — не видно) → 36 символов base64."""
    q = [255 if v < 0 else int(round(min(1.0, v) * 250)) for v in points]
    return base64.b64encode(bytes(q)).decode("ascii")


def unpack_pose(s: str) -> list[float]:
    return [-1.0 if b == 255 else b / 250 for b in base64.b64decode(s)]


def pack(vectors: Iterable[np.ndarray]) -> str:
    """Векторы → строка (float16, base64): компактно для JSON в git."""
    return base64.b64encode(np.stack(list(vectors)).astype(np.float16).tobytes()).decode("ascii")


def unpack(s: str, dim: int = 384) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32).reshape(-1, dim)
