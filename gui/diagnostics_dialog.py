import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from core.audio_capture import AudioCapture
from gui.audio_meter import AudioMeter


class DeviceTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio Device Test")
        self.setMinimumWidth(450)
        self._capture = AudioCapture()
        self._is_testing = False
        self._chunk_count = 0
        self._test_start = 0.0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info_group = QGroupBox("Available Audio Devices")
        info_layout = QVBoxLayout()

        self._device_list = QTextEdit()
        self._device_list.setReadOnly(True)
        self._device_list.setMaximumHeight(120)
        info_layout.addWidget(self._device_list)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        test_group = QGroupBox("Test Monitor Device")
        test_layout = QVBoxLayout()

        self._test_device_input = QLineEdit()
        self._test_device_input.setPlaceholderText("Monitor device name (auto-detected)")
        test_layout.addWidget(self._test_device_input)

        self._meter = AudioMeter()
        test_layout.addWidget(self._meter)

        status_row = QHBoxLayout()
        self._chunk_label = QLabel("Chunks: 0")
        status_row.addWidget(self._chunk_label)
        self._elapsed_label = QLabel("Elapsed: 0.0s / 3.0s")
        status_row.addWidget(self._elapsed_label)
        status_row.addStretch()
        test_layout.addLayout(status_row)

        btn_row = QHBoxLayout()
        self._test_btn = QPushButton("Start Test (3s)")
        self._test_btn.clicked.connect(self._on_test)
        btn_row.addWidget(self._test_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        test_layout.addLayout(btn_row)

        self._test_result = QLabel("")
        self._test_result.setWordWrap(True)
        test_layout.addWidget(self._test_result)

        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self._refresh_devices()

    def _refresh_devices(self):
        devices = self._capture.discover_devices()
        monitor = self._capture.get_default_monitor()
        lines = []
        for d in devices:
            marker = " <-- default monitor" if monitor and d.index == monitor.index else ""
            lines.append(f"[{d.index}] {d.name}{marker}")
        self._device_list.setPlainText("\n".join(lines) if lines else "No audio devices found")
        if monitor:
            self._test_device_input.setText(monitor.name)

    def _on_test(self):
        monitor = self._capture.get_default_monitor()
        if monitor is None:
            self._test_result.setText(
                "No monitor device found. Make sure PipeWire/PulseAudio is running "
                "and audio is playing in another application."
            )
            return

        self._is_testing = True
        self._chunk_count = 0
        self._test_start = time.time()
        self._test_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._test_result.setText("Recording 3 seconds of audio...")
        self._capture.set_audio_callback(self._on_audio)
        self._capture.set_level_callback(self._on_level)
        self._capture.start(monitor)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.start(100)
        QTimer.singleShot(3000, self._finish_test)

    def _update_elapsed(self):
        if self._is_testing:
            elapsed = time.time() - self._test_start
            self._elapsed_label.setText(f"Elapsed: {elapsed:.1f}s / 3.0s")

    def _finish_test(self):
        if self._is_testing:
            self._capture.stop()
            self._is_testing = False
            self._timer.stop()
            self._test_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._meter.set_level(0.0)
            if self._chunk_count > 0:
                self._test_result.setText(
                    f"Test complete — received {self._chunk_count} audio chunks successfully!"
                )
            else:
                self._test_result.setText(
                    "Test complete but no audio received. "
                    "Check that audio is playing and the monitor device is correct."
                )

    def _on_stop(self):
        if self._is_testing:
            self._capture.stop()
            self._is_testing = False
            self._timer.stop()
            self._test_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._meter.set_level(0.0)
            self._test_result.setText(f"Stopped — received {self._chunk_count} chunks")

    def _on_audio(self, data, rate):
        self._chunk_count += 1
        self._chunk_label.setText(f"Chunks: {self._chunk_count}")

    def _on_level(self, level):
        self._meter.set_level(level)

    def closeEvent(self, event):
        if self._is_testing:
            self._capture.stop()
        event.accept()


class EngineTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test STT Engine")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self._capture = AudioCapture()
        self._is_testing = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "Record a few seconds of audio to test the STT engine.\n"
            "Speak clearly or play audio from a browser tab."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_row = QHBoxLayout()
        self._test_vosk_btn = QPushButton("Test Vosk (streaming)")
        self._test_vosk_btn.clicked.connect(self._test_vosk)
        btn_row.addWidget(self._test_vosk_btn)
        self._test_whisper_btn = QPushButton("Test Whisper (batch)")
        self._test_whisper_btn.clicked.connect(self._test_whisper)
        btn_row.addWidget(self._test_whisper_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        self._result = QTextEdit()
        self._result.setReadOnly(True)
        self._result.setPlaceholderText("Test results will appear here...")
        layout.addWidget(self._result)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def _re_enable_buttons(self):
        self._stop_btn.setEnabled(False)
        self._test_vosk_btn.setEnabled(True)
        self._test_whisper_btn.setEnabled(True)

    def _test_vosk(self):
        from core.engine_vosk import VoskEngine

        self._result.clear()
        self._result.append("Testing Vosk streaming...")
        self._vosk = VoskEngine()
        self._vosk.partial_result.connect(lambda t: self._result.append(f"Partial: {t}"))
        self._vosk.final_result.connect(lambda t, ts: self._result.append(f"Final: {t}"))
        self._vosk.model_loaded.connect(
            lambda n: self._result.append(f"Vosk model loaded: {n}")
        )
        self._vosk.model_error.connect(
            lambda e: self._result.append(f"Vosk error: {e}")
        )

        def on_model_ready():
            self._result.append("Model ready, starting capture...")
            self._start_capture(lambda d, sr: self._vosk.feed_audio(d, sr))

        self._vosk.model_loaded.connect(lambda _: on_model_ready())
        self._vosk.load_model()

    def _test_whisper(self):
        from core.engine_whisper import WhisperEngine

        self._result.clear()
        self._result.append("Loading Whisper model...")
        self._whisper = WhisperEngine()

        def on_whisper_ready(name):
            self._result.append(f"Whisper model loaded: {name}")
            self._result.append("Recording 5 seconds of audio...")
            self._start_capture(lambda d, sr: self._whisper.feed_audio(d, sr))

            def finish_and_transcribe():
                if self._is_testing:
                    self._capture.stop()
                    self._is_testing = False
                    self._re_enable_buttons()
                    if self._whisper.is_loaded():
                        self._result.append("5s recorded, transcribing...")
                        self._whisper.transcribe()
                    else:
                        self._result.append("Whisper model not ready, cannot transcribe.")

            QTimer.singleShot(5000, finish_and_transcribe)

        self._whisper.transcription_ready.connect(
            lambda segs: self._result.append(
                "Whisper result: " + " ".join(s["text"] for s in segs)
            )
        )
        self._whisper.transcription_error.connect(
            lambda e: self._result.append(f"Whisper error: {e}")
        )
        self._whisper.model_loaded.connect(on_whisper_ready)
        self._whisper.load_model()

    def _start_capture(self, audio_callback):
        monitor = self._capture.get_default_monitor()
        if monitor is None:
            self._result.append(
                "No monitor device found. Make sure PipeWire/PulseAudio is running."
            )
            self._re_enable_buttons()
            return

        self._is_testing = True
        self._stop_btn.setEnabled(True)
        self._test_vosk_btn.setEnabled(False)
        self._test_whisper_btn.setEnabled(False)
        self._capture.set_audio_callback(audio_callback)
        self._capture.start(monitor)

    def _on_stop(self):
        if self._is_testing:
            self._capture.stop()
            self._is_testing = False
            self._re_enable_buttons()
            self._result.append("Stopped")

    def closeEvent(self, event):
        if self._is_testing:
            self._capture.stop()
        event.accept()
