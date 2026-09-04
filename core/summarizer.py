import threading
from pathlib import Path

from huggingface_hub import hf_hub_download
from PySide6.QtCore import QObject, Signal

_MODEL_CACHE_DIR = Path.home() / ".cache" / "speechTranscriptor" / "llm"

_MODELS = {
    "3b": {
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "display": "Qwen2.5-3B (~1.8GB, better quality)",
    },
    "1.5b": {
        "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "display": "Qwen2.5-1.5B (~1GB, faster)",
    },
}


class Summarizer(QObject):
    summary_ready = Signal(str)
    summary_error = Signal(str)
    model_loading = Signal()
    model_loaded = Signal()

    def __init__(self, model_key: str = "3b", parent=None):
        super().__init__(parent)
        self._model_key = model_key if model_key in _MODELS else "3b"
        self._llm = None
        self._loaded = False
        self._model_path: Path | None = None

    @staticmethod
    def available_models() -> dict:
        return dict(_MODELS)

    def current_model_key(self) -> str:
        return self._model_key

    def current_model_display(self) -> str:
        return _MODELS[self._model_key]["display"]

    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self):
        if self._loaded:
            return

        def _load():
            try:
                _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self.model_loading.emit()

                info = _MODELS[self._model_key]
                self._model_path = Path(
                    hf_hub_download(
                        repo_id=info["repo_id"],
                        filename=info["filename"],
                        local_dir=str(_MODEL_CACHE_DIR),
                    )
                )

                from llama_cpp import Llama

                self._llm = Llama(
                    model_path=str(self._model_path),
                    n_ctx=4096,
                    n_threads=4,
                    verbose=False,
                )
                self._loaded = True
                self.model_loaded.emit()
            except Exception as e:
                self.summary_error.emit(str(e))

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def switch_model(self, model_key: str):
        if model_key not in _MODELS or model_key == self._model_key:
            return
        self.cleanup()
        self._model_key = model_key
        self.load_model()

    def summarize(self, text: str):
        if not self._loaded or self._llm is None:
            self.summary_error.emit("LLM model not loaded yet")
            return

        def _summarize():
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that summarizes transcripts. "
                            "Use bullet points for key topics. "
                            "Be clear and concise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following transcript:\n\n{text}",
                    },
                ]

                output = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.3,
                    top_p=0.9,
                )

                summary = output["choices"][0]["message"]["content"].strip()
                self.summary_ready.emit(summary)
            except Exception as e:
                self.summary_error.emit(str(e))

        thread = threading.Thread(target=_summarize, daemon=True)
        thread.start()

    def cleanup(self):
        self._llm = None
        self._loaded = False
