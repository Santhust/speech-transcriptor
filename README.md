# Speech Transcriptor

A Python desktop application that captures system audio on Linux and transcribes it to text using local speech-to-text engines, with optional LLM-powered summarization.

## Features

- **System audio capture** — records any audio playing on your computer (browser tabs, music, meetings) via PipeWire/PulseAudio monitor sources
- **Microphone capture** — record from any microphone via the input-device dropdown in the toolbar
- **Streaming transcription** — real-time partial and final transcription using Vosk
- **Batch transcription** — record first, transcribe after using either Vosk or faster-whisper
- **Multi-language** — English and German recognition, plus Whisper auto-detect in batch mode
- **LLM summarization** — generate summaries of transcripts using local Qwen2.5 models (1.5B or 3B) via llama-cpp-python
- **Export** — save transcripts as .txt or .srt (subtitle) files
- **Search & highlight** — Ctrl+F search with keyword highlighting
- **Audio meter** — visual level meter during recording
- **Diagnostics** — built-in device and engine test dialogs

## Requirements

- Linux with PipeWire or PulseAudio
- Python 3.10+
- ~40MB disk for STT models (auto-downloaded on first run)
- ~1-2GB disk for LLM model (auto-downloaded on first summarization)

## Setup

```bash
git clone <repo-url>
cd speechTranscriptor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Models are downloaded automatically on first use, with live progress shown in the status bar:
- **Vosk** small-en-us (~40MB) or small-de (~40MB) — for streaming and batch transcription (depends on selected language)
- **faster-whisper** tiny int8 (~40MB) — alternative batch transcription engine (multilingual)
- **Qwen2.5** 1.5B or 3B GGUF (~1-2GB) — for LLM summarization

## Usage

1. Launch the app: `python main.py`
2. Pick an input device in the toolbar: a 🔊 monitor source captures system audio, a 🎤 microphone captures your voice (choice is remembered)
3. Press **Ctrl+R** or the toolbar Record button to start
4. Watch transcription appear in real-time (streaming mode) or press **Ctrl+S** to stop and transcribe (batch mode)
5. Use **Ctrl+U** or toolbar Summarize to generate an LLM summary

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+R | Start recording |
| Ctrl+S | Stop recording |
| Ctrl+P | Pause/Resume |
| Ctrl+E | Export transcript |
| Ctrl+U | Summarize transcript |
| Ctrl+F | Search in transcript |
| Ctrl+1 | Streaming mode |
| Ctrl+2 | Batch mode |
| Ctrl+L | Clear display |
| Ctrl+, | Preferences |

## Configuration

Edit preferences via Edit > Preferences (Ctrl+,):
- Recognition language (English, German, or auto-detect in batch mode with faster-whisper)
- Batch transcription engine (Vosk or faster-whisper)
- Whisper model size and compute type
- LLM model selection (3B or 1.5B)
- Output directory and default format
- Font size and highlight keywords

## License

This project uses PySide6 (LGPL). See individual dependency licenses for details.
