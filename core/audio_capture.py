import queue
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pulsectl

_TARGET_RATE = 16000
_CHUNK_FRAMES = 3200
_CHUNK_BYTES = _CHUNK_FRAMES * 2


@dataclass
class AudioDevice:
    index: int
    name: str
    display_name: str
    is_monitor: bool
    sample_rate: int
    channels: int
    pulse_source_name: str | None = None


class AudioCapture:
    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._processor_thread: threading.Thread | None = None
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._monitor_devices: list[AudioDevice] = []
        self._all_devices: list[AudioDevice] = []
        self._selected_device: AudioDevice | None = None
        self._recording = False
        self._paused = False
        self._audio_callback: Callable[[np.ndarray, int], None] | None = None
        self._level_callback: Callable[[float], None] | None = None

    def discover_devices(self) -> list[AudioDevice]:
        self._all_devices.clear()
        self._monitor_devices.clear()
        idx = 0

        try:
            pulse = pulsectl.Pulse("speech-transcriptor-discovery")
            for src in pulse.source_list():
                is_monitor = src.name.endswith(".monitor")
                dev = AudioDevice(
                    index=idx,
                    name=src.name,
                    display_name=src.description or src.name,
                    is_monitor=is_monitor,
                    sample_rate=src.sample_spec.rate if src.sample_spec else _TARGET_RATE,
                    channels=src.sample_spec.channels if src.sample_spec else 1,
                    pulse_source_name=src.name,
                )
                self._all_devices.append(dev)
                if is_monitor:
                    self._monitor_devices.append(dev)
                idx += 1
            pulse.close()
        except Exception:
            pass

        return self._all_devices

    def get_monitor_devices(self) -> list[AudioDevice]:
        if not self._monitor_devices:
            self.discover_devices()
        return self._monitor_devices

    def get_default_monitor(self) -> AudioDevice | None:
        monitors = self.get_monitor_devices()
        if not monitors:
            return None
        try:
            pulse = pulsectl.Pulse("speech-transcriptor-default")
            monitor_name = f"{pulse.server_info().default_sink_name}.monitor"
            pulse.close()
            for m in monitors:
                if m.pulse_source_name == monitor_name:
                    return m
        except Exception:
            pass
        return monitors[0]

    def select_device(self, device: AudioDevice):
        self._selected_device = device

    def get_selected_device(self) -> AudioDevice | None:
        return self._selected_device

    def set_audio_callback(self, callback: Callable[[np.ndarray, int], None]):
        self._audio_callback = callback

    def set_level_callback(self, callback: Callable[[float], None]):
        self._level_callback = callback

    def start(self, device: AudioDevice | None = None):
        if self._recording:
            return

        dev = device or self._selected_device
        if dev is None:
            raise RuntimeError("No audio device selected")
        if not dev.pulse_source_name:
            raise RuntimeError(f"Device '{dev.name}' has no PulseAudio source")

        cmd = [
            "parec",
            f"--device={dev.pulse_source_name}",
            "--format=s16le",
            f"--rate={_TARGET_RATE}",
            "--channels=1",
        ]
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )
        self._recording = True
        self._paused = False
        self._audio_queue = queue.Queue()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
        self._reader_thread.start()
        self._processor_thread.start()

    def _reader_loop(self):
        proc = self._process
        if proc is None or proc.stdout is None:
            self._audio_queue.put(None)
            return
        while self._recording:
            raw = proc.stdout.read(_CHUNK_BYTES)
            if not raw:
                break
            self._audio_queue.put(raw)
        self._audio_queue.put(None)

    def _processor_loop(self):
        while True:
            raw = self._audio_queue.get()
            if raw is None:
                break
            if self._paused:
                continue
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))
            level = min(rms * 10, 1.0)
            if self._level_callback:
                self._level_callback(level)
            if self._audio_callback:
                self._audio_callback(samples, _TARGET_RATE)

    def stop(self):
        self._recording = False
        self._paused = False

        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

        if self._processor_thread is not None:
            self._processor_thread.join(timeout=2)
            self._processor_thread = None

        if self._level_callback:
            self._level_callback(0.0)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def is_recording(self) -> bool:
        return self._recording

    def is_paused(self) -> bool:
        return self._paused
