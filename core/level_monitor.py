import subprocess
import threading

import numpy as np

_CHUNK_FRAMES = 3200
_CHUNK_BYTES = _CHUNK_FRAMES * 2
_TARGET_RATE = 16000


class LevelMonitor:
    def __init__(self, source_name: str, callback, parent=None):
        self._source_name = source_name
        self._callback = callback
        self._parent = parent
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._process = subprocess.Popen(
            [
                "parec",
                f"--device={self._source_name}",
                "--format=s16le",
                f"--rate={_TARGET_RATE}",
                "--channels=1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
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
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._callback(0.0)

    def is_running(self) -> bool:
        return self._running

    def _loop(self):
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        while self._running:
            raw = proc.stdout.read(_CHUNK_BYTES)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))
            self._callback(min(rms * 10, 1.0))
