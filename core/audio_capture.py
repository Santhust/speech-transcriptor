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


class _SourceState:
    def __init__(self):
        self.buf = np.empty(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.active = True


class AudioCapture:
    def __init__(self):
        self._processes: list[subprocess.Popen] = []
        self._reader_threads: list[threading.Thread] = []
        self._processor_threads: list[threading.Thread] = []
        self._queues: list[queue.Queue] = []
        self._states: list[_SourceState] = []
        self._sources: list[AudioDevice] = []
        self._mixer_thread: threading.Thread | None = None
        self._dual_mode: str | None = None
        self._monitor_devices: list[AudioDevice] = []
        self._all_devices: list[AudioDevice] = []
        self._selected_device: AudioDevice | None = None
        self._recording = False
        self._paused = False
        self._audio_callback: Callable[..., None] | None = None
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

    def get_all_devices(self) -> list[AudioDevice]:
        if not self._all_devices:
            self.discover_devices()
        return list(self._all_devices)

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

    def set_audio_callback(self, callback: Callable[..., None]):
        self._audio_callback = callback

    def set_level_callback(self, callback: Callable[[float], None]):
        self._level_callback = callback

    def start(self, device: AudioDevice | None = None):
        dev = device or self._selected_device
        self._begin([dev], mode=None)

    def start_dual(self, mic_device: AudioDevice, system_device: AudioDevice, mode: str):
        if mode not in ("merged", "separate"):
            raise ValueError(f"Unknown dual mode: {mode}")
        self._begin([system_device, mic_device], mode=mode)

    def _begin(self, devices: list[AudioDevice | None], mode: str | None):
        if self._recording:
            return

        for dev in devices:
            if dev is None:
                raise RuntimeError("No audio device selected")
            if not dev.pulse_source_name:
                raise RuntimeError(f"Device '{dev.name}' has no PulseAudio source")

        self._dual_mode = mode
        self._recording = True
        self._paused = False
        self._processes = []
        self._reader_threads = []
        self._processor_threads = []
        self._queues = []
        self._states = []
        self._sources = []

        for dev in devices:
            cmd = [
                "parec",
                f"--device={dev.pulse_source_name}",
                "--format=s16le",
                f"--rate={_TARGET_RATE}",
                "--channels=1",
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
            )
            q: queue.Queue = queue.Queue()
            state = _SourceState()
            self._processes.append(proc)
            self._queues.append(q)
            self._states.append(state)
            self._sources.append(dev)

            reader = threading.Thread(target=self._reader_loop, args=(proc, q), daemon=True)
            processor = threading.Thread(
                target=self._processor_loop,
                args=(q, state, len(self._sources) - 1),
                daemon=True,
            )
            self._reader_threads.append(reader)
            self._processor_threads.append(processor)

        for t in self._reader_threads + self._processor_threads:
            t.start()

        if mode == "merged":
            self._mixer_thread = threading.Thread(target=self._mixer_loop, daemon=True)
            self._mixer_thread.start()

    def _reader_loop(self, proc: subprocess.Popen, q: queue.Queue):
        stream = proc.stdout
        if stream is None:
            q.put(None)
            return
        while self._recording:
            raw = stream.read(_CHUNK_BYTES)
            if not raw:
                break
            q.put(raw)
        q.put(None)

    def _processor_loop(self, q: queue.Queue, state: _SourceState, source_idx: int):
        while True:
            raw = q.get()
            if raw is None:
                with state.lock:
                    state.active = False
                break
            if self._paused:
                continue
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            if self._dual_mode == "separate":
                if self._audio_callback:
                    self._audio_callback(samples, _TARGET_RATE, source_idx)
                continue

            if self._dual_mode == "merged":
                with state.lock:
                    state.buf = np.concatenate([state.buf, samples])
                continue

            rms = float(np.sqrt(np.mean(samples ** 2)))
            level = min(rms * 10, 1.0)
            if self._level_callback:
                self._level_callback(level)
            if self._audio_callback:
                self._audio_callback(samples, _TARGET_RATE)

    def _mixer_loop(self):
        silence = np.zeros(_CHUNK_FRAMES, dtype=np.float32)
        while self._recording:
            avail = _CHUNK_FRAMES
            for s in self._states:
                with s.lock:
                    avail = min(avail, len(s.buf))
            if avail >= _CHUNK_FRAMES:
                chunks = []
                for s in self._states:
                    with s.lock:
                        take = min(avail, len(s.buf))
                        chunk = s.buf[:take].copy()
                        s.buf = s.buf[take:]
                        if len(chunk) < _CHUNK_FRAMES:
                            chunk = np.concatenate(
                                [chunk, np.zeros(_CHUNK_FRAMES - len(chunk), dtype=np.float32)]
                            )
                    chunks.append(chunk)
                mixed = np.mean(chunks, axis=0)
                mixed = np.clip(mixed, -1.0, 1.0)
                if self._audio_callback:
                    self._audio_callback(mixed, _TARGET_RATE)
            else:
                if all(not s.active for s in self._states) and avail < _CHUNK_FRAMES:
                    break
                threading.Event().wait(0.01)

    def stop(self):
        self._recording = False
        self._paused = False

        for proc in self._processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._processes = []

        for t in self._reader_threads:
            t.join(timeout=2)
        for t in self._processor_threads:
            t.join(timeout=2)
        if self._mixer_thread is not None:
            self._mixer_thread.join(timeout=2)
        self._reader_threads = []
        self._processor_threads = []
        self._mixer_thread = None
        self._queues = []
        self._states = []
        self._dual_mode = None

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
