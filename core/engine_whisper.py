import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal


class WhisperEngine(QObject):
    transcription_ready = Signal(list)
    transcription_error = Signal(str)
    model_loaded = Signal(str)
    download_progress = Signal(int, int)

    def __init__(
        self,
        model_size: str = "tiny",
        compute_type: str = "int8",
        language: str | None = "en",
        parent=None,
    ):
        super().__init__(parent)
        self._model_size = model_size
        self._compute_type = compute_type
        self._language = language
        self._model = None
        self._loaded = False
        self._audio_buffer: list[np.ndarray] = []
        self._sample_rate = 16000

    def load_model(self):
        if self._loaded:
            return

        def _load():
            try:
                from core.model_downloader import ensure_whisper_model

                try:
                    ensure_whisper_model(
                        self._model_size,
                        progress_cb=lambda done, total: self.download_progress.emit(
                            done, total
                        ),
                    )
                except Exception as e:
                    self.transcription_error.emit(f"Model download failed: {e}")
                    return
                self.download_progress.emit(0, 0)

                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type=self._compute_type,
                )
                self._loaded = True
                self.model_loaded.emit(self._model_size)
            except Exception as e:
                self.transcription_error.emit(str(e))

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def is_loaded(self) -> bool:
        return self._loaded

    def set_sample_rate(self, rate: int):
        self._sample_rate = rate

    def feed_audio(self, audio_data: np.ndarray, sample_rate: int):
        self._sample_rate = sample_rate
        self._audio_buffer.append(audio_data.copy())

    def clear_buffer(self):
        self._audio_buffer.clear()

    def get_buffer_duration(self) -> float:
        total_samples = sum(len(a) for a in self._audio_buffer)
        return total_samples / self._sample_rate if self._sample_rate > 0 else 0

    def transcribe(self):
        if not self._loaded or not self._audio_buffer:
            self.transcription_ready.emit([])
            return

        def _transcribe():
            try:
                audio = np.concatenate(self._audio_buffer)
                audio_float32 = audio.astype(np.float32)

                segments, info = self._model.transcribe(
                    audio_float32,
                    beam_size=5,
                    language=self._language,
                    vad_filter=True,
                )

                results = []
                for seg in segments:
                    results.append({
                        "text": seg.text.strip(),
                        "start": seg.start,
                        "end": seg.end,
                    })

                self.transcription_ready.emit(results)
            except Exception as e:
                self.transcription_error.emit(str(e))

        thread = threading.Thread(target=_transcribe, daemon=True)
        thread.start()

    def transcribe_file(self, file_path: str):
        if not self._loaded:
            self.transcription_error.emit("Model not loaded")
            return

        def _transcribe():
            try:
                segments, info = self._model.transcribe(
                    file_path,
                    beam_size=5,
                    language=self._language,
                    vad_filter=True,
                )

                results = []
                for seg in segments:
                    results.append({
                        "text": seg.text.strip(),
                        "start": seg.start,
                        "end": seg.end,
                    })

                self.transcription_ready.emit(results)
            except Exception as e:
                self.transcription_error.emit(str(e))

        thread = threading.Thread(target=_transcribe, daemon=True)
        thread.start()

    def cleanup(self):
        self._model = None
        self._loaded = False
        self._audio_buffer.clear()
