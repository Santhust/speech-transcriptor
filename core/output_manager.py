import os
from datetime import datetime
from pathlib import Path

import srt
from PySide6.QtCore import QObject, Signal


class OutputManager(QObject):
    export_done = Signal(str)

    def __init__(self, output_dir: str | None = None, parent=None):
        super().__init__(parent)
        self._output_dir = Path(output_dir) if output_dir else (
            Path.home() / "SpeechTranscriptor" / "output"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def set_output_dir(self, path: str):
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_dir(self) -> Path:
        return self._output_dir

    def save_txt(self, segments: list[dict], filename: str | None = None) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.txt"

        path = self._output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for seg in segments:
                start = seg.get("start", 0)
                text = seg.get("text", "")
                minutes = int(start // 60)
                seconds = int(start % 60)
                f.write(f"[{minutes:02d}:{seconds:02d}] {text}\n")

        self.export_done.emit(str(path))
        return path

    def save_srt(self, segments: list[dict], filename: str | None = None) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.srt"

        path = self._output_dir / filename
        srt_segments = []
        for i, seg in enumerate(segments, 1):
            start = seg.get("start", 0)
            end = seg.get("end", start + 2)
            text = seg.get("text", "")

            start_td = _seconds_to_timedelta(start)
            end_td = _seconds_to_timedelta(end)

            srt_segments.append(srt.Subtitle(
                index=i,
                start=start_td,
                end=end_td,
                content=text,
            ))

        with open(path, "w", encoding="utf-8") as f:
            f.write(srt.compose(srt_segments))

        self.export_done.emit(str(path))
        return path

    def save_transcript(self, segments: list[dict], fmt: str = "txt") -> Path:
        if fmt == "srt":
            return self.save_srt(segments)
        return self.save_txt(segments)

    def format_plain_text(self, segments: list[dict], show_timestamps: bool = True) -> str:
        lines = []
        for seg in segments:
            start = seg.get("start", 0)
            text = seg.get("text", "")
            if show_timestamps:
                minutes = int(start // 60)
                seconds = int(start % 60)
                lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")
            else:
                lines.append(text)
        return "\n".join(lines)


def _seconds_to_timedelta(seconds: float):
    from datetime import timedelta
    return timedelta(seconds=seconds)
