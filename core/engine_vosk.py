import json
import threading
import time
from pathlib import Path

import numpy as np
import vosk
from PySide6.QtCore import QObject, Signal

_MODEL_CACHE_DIR = Path.home() / ".cache" / "speechTranscriptor" / "vosk"


class VoskEngine(QObject):
    partial_result = Signal(str)
    final_result = Signal(str, float)
    transcription_ready = Signal(list)
    model_loaded = Signal(str)
    model_error = Signal(str)

    def __init__(self, model_name: str = "vosk-model-small-en-us-0.15", parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._model: vosk.Model | None = None
        self._recognizer: vosk.KaldiRecognizer | None = None
        self._sample_rate = 16000
        self._loaded = False
        self._lock = threading.Lock()
        self._audio_buffer: list[np.ndarray] = []

    def load_model(self):
        if self._loaded:
            return

        def _load():
            try:
                _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._model = vosk.Model(model_name=self._model_name)
                self._recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
                self._recognizer.SetWords(True)
                self._loaded = True
                self.model_loaded.emit(self._model_name)
            except Exception as e:
                self.model_error.emit(str(e))

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def is_loaded(self) -> bool:
        return self._loaded

    def set_sample_rate(self, rate: int):
        self._sample_rate = rate
        if self._recognizer:
            self._recognizer = vosk.KaldiRecognizer(self._model, rate)

    def feed_audio(self, audio_data: np.ndarray, sample_rate: int):
        if not self._loaded or self._recognizer is None:
            return

        pcm_data = (audio_data * 32767).astype(np.int16).tobytes()

        with self._lock:
            if self._recognizer.AcceptWaveform(pcm_data):
                result = json.loads(self._recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    self.final_result.emit(text, time.time())
            else:
                partial = json.loads(self._recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                if text:
                    self.partial_result.emit(text)

    def feed_audio_buffered(self, audio_data: np.ndarray, sample_rate: int):
        self._sample_rate = sample_rate
        self._audio_buffer.append(audio_data.copy())

    def clear_buffer(self):
        self._audio_buffer.clear()

    def get_buffer_duration(self) -> float:
        total_samples = sum(len(a) for a in self._audio_buffer)
        return total_samples / self._sample_rate if self._sample_rate > 0 else 0

    def transcribe_batch(self):
        if not self._loaded or not self._audio_buffer:
            self.transcription_ready.emit([])
            return

        def _transcribe():
            try:
                audio = np.concatenate(self._audio_buffer)
                pcm_data = (audio * 32767).astype(np.int16).tobytes()

                rec = vosk.KaldiRecognizer(self._model, self._sample_rate)
                rec.SetWords(True)

                chunk_size = self._sample_rate * 2
                results = []
                time_offset = 0.0

                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i:i + chunk_size]
                    if rec.AcceptWaveform(chunk):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip()
                        if text:
                            chunk_frames = len(chunk) // 2
                            results.append({
                                "text": text,
                                "start": time_offset,
                                "end": time_offset + chunk_frames / self._sample_rate,
                            })
                    chunk_frames = len(chunk) // 2
                    time_offset += chunk_frames / self._sample_rate

                final = json.loads(rec.FinalResult())
                final_text = final.get("text", "").strip()
                if final_text:
                    results.append({
                        "text": final_text,
                        "start": time_offset,
                        "end": time_offset + 1.0,
                    })

                self.transcription_ready.emit(results)
            except Exception as e:
                self.model_error.emit(str(e))

        thread = threading.Thread(target=_transcribe, daemon=True)
        thread.start()

    def get_final(self) -> str:
        if not self._loaded or self._recognizer is None:
            return ""
        with self._lock:
            result = json.loads(self._recognizer.FinalResult())
            return result.get("text", "").strip()

    def reset(self):
        if self._model and self._sample_rate:
            with self._lock:
                self._recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
                self._recognizer.SetWords(True)

    def cleanup(self):
        with self._lock:
            self._recognizer = None
            self._model = None
            self._loaded = False
        self._audio_buffer.clear()
