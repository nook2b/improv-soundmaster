"""Аудиоустройства: вывод (микшер → колонки) и ввод (профиль входа → запись и, позже, AI).

`sounddevice` импортируется лениво: логика и тесты работают без PortAudio. На macOS колёса
sounddevice уже содержат PortAudio.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

from ..clock import Clock
from .mixer import Mixer

BlockCallback = Callable[[np.ndarray, float], None]  # блок, оценка времени выхода/захвата по Clock


def _sd():
    try:
        import sounddevice as sd  # type: ignore
    except OSError as e:  # PortAudio не найден
        raise RuntimeError(f"sounddevice недоступен: {e}") from e
    return sd


def list_devices() -> str:
    return str(_sd().query_devices())


def find_device(name: str | None, kind: str) -> int | None:
    """Найти устройство по подстроке имени. None — устройство по умолчанию."""
    if not name:
        return None
    sd = _sd()
    key = "max_output_channels" if kind == "output" else "max_input_channels"
    matches = [i for i, d in enumerate(sd.query_devices()) if name.lower() in d["name"].lower() and d[key] > 0]
    if not matches:
        raise RuntimeError(f"устройство {kind} {name!r} не найдено. Список: python -m soundmaster devices")
    return matches[0]


class SoundDeviceOutput:
    def __init__(
        self,
        mixer: Mixer,
        clock: Clock,
        device: str | None = None,
        blocksize: int = 256,
        latency: str | float = "low",
        on_block: BlockCallback | None = None,
    ):
        self.mixer = mixer
        self.clock = clock
        self.device = device
        self.blocksize = blocksize
        self.latency = latency
        self.on_block = on_block
        self.output_latency = 0.0
        self.xruns = 0
        # Ошибки в callback: исключение остановило бы поток PortAudio до конца шоу. В callback
        # их только считаем и запоминаем (без журнала), сообщает о них основной цикл.
        self.render_errors = 0
        self.last_error: BaseException | None = None
        self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1
        try:
            block = self.mixer.render(frames)
            outdata[:] = block
        except Exception as e:  # noqa: BLE001 — блок тишины вместо обрыва вывода
            self.render_errors += 1
            self.last_error = e
            outdata.fill(0)
            return
        if self.on_block:
            try:
                dac_delay = max(0.0, time_info.outputBufferDacTime - time_info.currentTime)
                self.on_block(block, self.clock.now() + dac_delay)
            except Exception as e:  # noqa: BLE001 — сбой записи не должен глушить звук
                self.render_errors += 1
                self.last_error = e

    def take_error(self) -> tuple[int, str | None]:
        """Для журнала из основного цикла: (сколько ошибок всего, текст последней) и сброс последней."""
        e, self.last_error = self.last_error, None
        return self.render_errors, (f"{type(e).__name__}: {e}" if e else None)

    def start(self) -> None:
        sd = _sd()
        self._stream = sd.OutputStream(
            samplerate=self.mixer.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self.blocksize,
            latency=self.latency,
            device=find_device(self.device, "output"),
            callback=self._callback,
        )
        self._stream.start()
        self.output_latency = float(self._stream.latency)

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class NullOutput:
    """Вывод без звуковой карты: микшер считается в реальном темпе в отдельном потоке."""

    def __init__(self, mixer: Mixer, clock: Clock, blocksize: int = 256, on_block: BlockCallback | None = None):
        self.mixer = mixer
        self.clock = clock
        self.blocksize = blocksize
        self.on_block = on_block
        self.output_latency = 0.0
        self.xruns = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        period = self.blocksize / self.mixer.sample_rate
        next_t = time.monotonic()
        while not self._stop.is_set():
            block = self.mixer.render(self.blocksize)
            if self.on_block:
                self.on_block(block, self.clock.now())
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="null-output", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


class SoundDeviceInput:
    """Захват входа по профилю. Callback только кладёт блоки в очередь."""

    def __init__(self, profile, clock: Clock, blocksize: int = 1024):
        self.profile = profile
        self.clock = clock
        self.blocksize = blocksize
        self.blocks: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue(maxsize=2000)
        self.overflows = 0
        self.dropped = 0
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.overflows += 1
        t = self.clock.now() - max(0.0, time_info.currentTime - time_info.inputBufferAdcTime)
        try:
            self.blocks.put_nowait((indata.copy(), t))
        except queue.Full:
            self.dropped += 1

    def start(self) -> None:
        sd = _sd()
        self._stream = sd.InputStream(
            samplerate=self.profile.sample_rate,
            channels=self.profile.device_channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=find_device(self.profile.device, "input"),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
