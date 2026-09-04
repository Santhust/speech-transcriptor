import os
from pathlib import Path

from PySide6.QtCore import QSettings


LANGUAGE_MODELS = {
    "en": ("English", "vosk-model-small-en-us-0.15"),
    "de": ("Deutsch (German)", "vosk-model-small-de-0.15"),
    "auto": ("Auto-detect (Whisper batch only)", "vosk-model-small-en-us-0.15"),
}


_DEFAULTS = {
    "audio/device_index": -1,
    "audio/device_name": "",
    "recognition/language": "en",
    "engine/streaming": "vosk",
    "engine/batch": "faster-whisper",
    "engine/summarizer": "qwen2.5-3b",
    "llm/model": "3b",
    "model/vosk_model": "vosk-model-small-en-us-0.15",
    "model/whisper_model": "tiny",
    "model/whisper_compute": "int8",
    "output/directory": str(Path.home() / "SpeechTranscriptor" / "output"),
    "output/auto_save": True,
    "output/format": "txt",
    "appearance/theme": "system",
    "appearance/font_size": 12,
    "appearance/keywords": "",
    "view/auto_scroll": True,
    "view/show_timestamps": True,
    "view/show_meter": False,
    "view/meter_system": False,
    "view/meter_mic": False,
}


class Config:
    def __init__(self):
        self._settings = QSettings("SpeechTranscriptor", "SpeechTranscriptor")

    def get(self, key: str):
        default = _DEFAULTS.get(key)
        return self._settings.value(key, default)

    def set(self, key: str, value):
        self._settings.setValue(key, value)

    def get_int(self, key: str) -> int:
        return int(self.get(key))

    def get_bool(self, key: str) -> bool:
        val = self.get(key)
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")

    def get_str(self, key: str) -> str:
        return str(self.get(key))

    def ensure_output_dir(self):
        path = Path(self.get_str("output/directory"))
        path.mkdir(parents=True, exist_ok=True)
        return path


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
