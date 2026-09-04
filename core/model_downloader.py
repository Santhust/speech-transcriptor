import json
import shutil
import threading
import zipfile
from pathlib import Path

import requests

_MODEL_LIST_URL = "https://alphacephei.com/vosk/models/model-list.json"
_MODELS_BASE_URL = "https://alphacephei.com/vosk/models/"
_VOSK_CACHE_DIR = Path.home() / ".cache" / "vosk"


def ensure_whisper_model(model_size: str, progress_cb=None) -> str:
    from huggingface_hub import snapshot_download

    repo_id = f"Systran/faster-whisper-{model_size}"
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return repo_id
    except Exception:
        pass

    if progress_cb is None:
        snapshot_download(repo_id=repo_id)
        return repo_id

    tqdm_cls = _make_progress_tqdm(progress_cb)
    snapshot_download(repo_id=repo_id, tqdm_class=tqdm_cls)
    return repo_id


def _make_progress_tqdm(progress_cb):
    from huggingface_hub.utils import tqdm as hub_tqdm

    lock = threading.Lock()
    entries: dict[int, list] = {}

    class ProgressTqdm(hub_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with lock:
                entries[id(self)] = [0, getattr(self, "total", 0) or 0]

        def update(self, n=1):
            displayed = super().update(n)
            done_total = None
            with lock:
                entry = entries.get(id(self))
                if entry is not None:
                    entry[0] += n
                    entry[1] = getattr(self, "total", 0) or 0
                    total = sum(e[1] for e in entries.values())
                    if total > 0:
                        done_total = (min(sum(e[0] for e in entries.values()), total), total)
            if done_total:
                progress_cb(*done_total)
            return displayed

        def close(self):
            with lock:
                entries.pop(id(self), None)
            return super().close()

    return ProgressTqdm



def ensure_vosk_model(model_name: str, progress_cb=None) -> Path:
    model_dir = _VOSK_CACHE_DIR / model_name
    if model_dir.is_dir() and any(model_dir.iterdir()):
        return model_dir

    url = _resolve_model_url(model_name)
    _VOSK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _VOSK_CACHE_DIR / f"{model_name}.zip"

    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with zip_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            tops = {n.split("/")[0] for n in archive.namelist()}
            top = next(iter(tops))
            archive.extractall(_VOSK_CACHE_DIR)
        extracted = _VOSK_CACHE_DIR / top
        if extracted.resolve() != model_dir.resolve():
            if model_dir.exists():
                shutil.rmtree(model_dir)
            extracted.rename(model_dir)
    finally:
        zip_path.unlink(missing_ok=True)

    return model_dir


def _resolve_model_url(model_name: str) -> str:
    try:
        resp = requests.get(_MODEL_LIST_URL, timeout=10)
        resp.raise_for_status()
        for entry in resp.json():
            if (
                isinstance(entry, dict)
                and entry.get("name") == model_name
                and entry.get("obsolete") != "true"
            ):
                return _MODELS_BASE_URL + f"{model_name}.zip"
    except Exception:
        pass
    return _MODELS_BASE_URL + f"{model_name}.zip"
