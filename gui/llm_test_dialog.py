import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from core.summarizer import Summarizer

_DUMMY_TEXT = (
    "The quarterly meeting discussed budget allocations for the marketing department. "
    "Sarah presented the new advertising campaign targeting young adults. "
    "Revenue projections show a fifteen percent increase by Q3. "
    "The team agreed to hire two additional content creators. "
    "Next meeting scheduled for Friday to review progress on social media strategy."
)


class LlmTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test LLM Summarization")
        self.setMinimumSize(600, 500)
        self._summarizer: Summarizer | None = None
        self._start_time = 0.0
        self._running = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        for key, info in Summarizer.available_models().items():
            self._model_combo.addItem(info["display"], key)
        model_row.addWidget(self._model_combo, 1)
        layout.addLayout(model_row)

        input_group = QGroupBox("Input Text (editable)")
        input_layout = QVBoxLayout()
        self._input = QTextEdit()
        self._input.setPlainText(_DUMMY_TEXT)
        self._input.setMinimumHeight(100)
        input_layout.addWidget(self._input)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        control_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Test")
        self._run_btn.clicked.connect(self._on_run)
        control_row.addWidget(self._run_btn)
        self._elapsed_label = QLabel("")
        control_row.addWidget(self._elapsed_label)
        control_row.addStretch()
        layout.addLayout(control_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        result_group = QGroupBox("Result")
        result_layout = QVBoxLayout()
        self._result = QTextEdit()
        self._result.setReadOnly(True)
        self._result.setPlaceholderText("Summary will appear here...")
        result_layout.addWidget(self._result)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _on_run(self):
        model_key = self._model_combo.currentData()
        text = self._input.toPlainText().strip()
        if not text:
            self._status_label.setText("No input text to summarize.")
            return

        self._run_btn.setEnabled(False)
        self._result.clear()
        self._status_label.setText("Loading model...")
        self._running = True
        self._start_time = time.time()
        self._timer.start(200)

        self._summarizer = Summarizer(model_key=model_key)
        self._summarizer.model_loaded.connect(lambda: self._on_model_ready(text))
        self._summarizer.model_loading.connect(
            lambda: self._status_label.setText("Downloading/loading model...")
        )
        self._summarizer.summary_ready.connect(self._on_result)
        self._summarizer.summary_error.connect(self._on_error)
        self._summarizer.load_model()

    def _on_model_ready(self, text: str):
        self._status_label.setText("Model ready — generating summary...")
        self._summarizer.summarize(text)

    def _on_result(self, summary: str):
        elapsed = time.time() - self._start_time
        self._timer.stop()
        self._running = False
        self._result.setPlainText(summary)
        self._status_label.setText(f"Done in {elapsed:.1f}s")
        self._elapsed_label.setText(f"{elapsed:.1f}s")
        self._run_btn.setEnabled(True)

    def _on_error(self, error: str):
        elapsed = time.time() - self._start_time
        self._timer.stop()
        self._running = False
        self._status_label.setText(f"Error ({elapsed:.1f}s): {error}")
        self._run_btn.setEnabled(True)

    def _tick(self):
        if self._running:
            elapsed = time.time() - self._start_time
            self._elapsed_label.setText(f"{elapsed:.1f}s")

    def closeEvent(self, event):
        self._timer.stop()
        if self._summarizer:
            self._summarizer.cleanup()
        event.accept()
