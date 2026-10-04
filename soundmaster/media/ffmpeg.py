"""Чтение звука и кадров через ffmpeg: из файла (прогон записей) и с камеры (macOS / Windows / Linux).

ffmpeg на Mac: `brew install ffmpeg`. Кадры приходят как RGB-массивы маленького размера:
для детектора движения и для отправки модели этого достаточно.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


class FFmpegError(RuntimeError):
    pass


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(f"{name} не найден. На Mac: brew install ffmpeg")
    return path


@dataclass
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool
    width: int = 0
    height: int = 0


def probe(path: str | Path) -> MediaInfo:
    out = subprocess.run(
        [ffmpeg_bin("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe не смог прочитать {path}: {out.stderr.strip()[:300]}")
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
    return MediaInfo(
        duration=float(data.get("format", {}).get("duration") or 0.0),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        has_video=video is not None,
        width=int(video.get("width", 0)) if video else 0,
        height=int(video.get("height", 0)) if video else 0,
    )


def scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """Размер после масштабирования до target_width с чётной высотой."""
    h = int(round(height * target_width / width / 2)) * 2
    return target_width, max(2, h)


def read_audio_blocks(path: str | Path, sample_rate: int = 16000, block: int = 1600) -> Iterator[np.ndarray]:
    """Моно float32 из любого файла (видео или аудио), блоками по `block` отсчётов."""
    proc = subprocess.Popen(
        [ffmpeg_bin(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-f", "f32le", "-"],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    nbytes = block * 4
    try:
        while True:
            raw = proc.stdout.read(nbytes)
            if not raw:
                break
            yield np.frombuffer(raw[: len(raw) // 4 * 4], np.float32)
    finally:
        proc.kill()
        proc.wait()


def _frame_args(fps: float, width: int, height: int) -> list[str]:
    return ["-vf", f"fps={fps},scale={width}:{height}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]


def read_video_frames(path: str | Path, fps: float = 5.0, width: int = 320) -> Iterator[tuple[float, np.ndarray]]:
    """(время кадра в секундах, RGB uint8 H×W×3) из файла с заданной частотой."""
    info = probe(path)
    if not info.has_video:
        return
    w, h = scaled_size(info.width, info.height, width)
    # На Mac — аппаратный декодер (4K HEVC с камеры/телефона), на других системах — программный.
    hw = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []
    proc = subprocess.Popen([ffmpeg_bin(), "-v", "error", *hw, "-i", str(path), "-an",
                             *_frame_args(fps, w, h)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    size = w * h * 3
    i = 0
    try:
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            yield i / fps, np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.kill()
        proc.wait()


def camera_input_args(camera: str, capture_size: str, capture_fps: int) -> list[str]:
    """Аргументы ffmpeg для живой камеры: macOS — avfoundation, Windows — DirectShow, Linux — v4l2."""
    if sys.platform == "darwin":
        return ["-f", "avfoundation", "-framerate", str(capture_fps), "-video_size", capture_size,
                "-i", f"{camera}:none"]
    if sys.platform == "win32":
        return ["-f", "dshow", "-framerate", str(capture_fps), "-video_size", capture_size,
                "-i", f"video={camera}"]
    dev = camera if camera.startswith("/dev/") else f"/dev/video{camera if camera.isdigit() else 0}"
    return ["-f", "v4l2", "-framerate", str(capture_fps), "-video_size", capture_size, "-i", dev]


def list_cameras() -> str:
    """Список камер (имена для config: video.camera)."""
    if sys.platform == "darwin":
        cmd = [ffmpeg_bin(), "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    elif sys.platform == "win32":
        cmd = [ffmpeg_bin(), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    else:
        return "Linux: камеры — /dev/video0, /dev/video1 … (укажите путь или номер в video.camera)"
    return subprocess.run(cmd, capture_output=True, text=True).stderr


class CameraReader:
    """Живая камера через ffmpeg (macOS, Windows, Linux). Для каждого кадра вызывает on_frame.

    На реальной камере ещё не проверено — проверить с Logitech Brio.
    """

    def __init__(self, camera: str, fps: float, width: int, height: int, on_frame, clock,
                 capture_size: str = "1280x720", capture_fps: int = 30):
        self.camera = camera
        self.fps = fps
        self.width = width
        self.height = height
        self.on_frame = on_frame
        self.clock = clock
        self.capture_size = capture_size
        self.capture_fps = capture_fps
        self.error: str | None = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        cmd = [ffmpeg_bin(), "-v", "error", *camera_input_args(self.camera, self.capture_size, self.capture_fps),
               *_frame_args(self.fps, self.width, self.height)]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._proc and self._proc.stdout
        size = self.width * self.height * 3
        while True:
            raw = self._proc.stdout.read(size)
            if len(raw) < size:
                err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
                self.error = f"камера остановилась: {err.strip()[:300]}"
                return
            self.on_frame(self.clock.now(), np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3))

    def stop(self) -> None:
        if self._proc:
            self._proc.kill()
            self._proc.wait()
