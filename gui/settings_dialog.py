from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.summarizer import Summarizer
from settings.config import get_config


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(480)
        self._cfg = get_config()
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        llm_group = QGroupBox("LLM Summarization")
        llm_form = QFormLayout()

        self._llm_model_combo = QComboBox()
        for key, info in Summarizer.available_models().items():
            self._llm_model_combo.addItem(info["display"], key)

        llm_form.addRow("Model:", self._llm_model_combo)
        llm_group.setLayout(llm_form)
        layout.addWidget(llm_group)

        engine_group = QGroupBox("Transcription Engine")
        engine_form = QFormLayout()

        self._whisper_model_label = QLabel("Whisper model size:")
        self._whisper_compute_label = QLabel("Whisper compute type:")

        self._whisper_model_input = QSpinBox()
        self._whisper_model_input.setRange(0, 100)
        self._whisper_model_input.setToolTip(
            "Whisper model: 0=tiny, 1=base, 2=small, 3=medium, 4=large"
        )

        self._whisper_compute_input = QLineEdit()
        self._whisper_compute_input.setPlaceholderText("int8")

        engine_form.addRow(
            QLabel("Batch engine:"), QLabel("Vosk or faster-whisper (set via View menu)")
        )
        engine_form.addRow(self._whisper_model_label, self._whisper_model_input)
        engine_form.addRow(self._whisper_compute_label, self._whisper_compute_input)
        engine_group.setLayout(engine_form)
        layout.addWidget(engine_group)

        output_group = QGroupBox("Output")
        output_form = QFormLayout()

        self._output_dir_input = QLineEdit()
        self._output_dir_input.setPlaceholderText(str(self._cfg.get("output/directory")))

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self._output_dir_input)
        dir_row.addWidget(browse_btn)

        self._auto_save_input = QLineEdit()
        self._auto_save_input.setPlaceholderText("txt or srt")

        output_form.addRow("Output directory:", dir_row)
        output_form.addRow("Default format:", self._auto_save_input)
        output_group.setLayout(output_form)
        layout.addWidget(output_group)

        appearance_group = QGroupBox("Appearance")
        appear_form = QFormLayout()

        self._font_size_input = QSpinBox()
        self._font_size_input.setRange(8, 24)
        self._font_size_input.setValue(12)

        self._keywords_input = QLineEdit()
        self._keywords_input.setPlaceholderText("comma-separated keywords to highlight")

        appear_form.addRow("Font size:", self._font_size_input)
        appear_form.addRow("Highlight keywords:", self._keywords_input)
        appearance_group.setLayout(appear_form)
        layout.addWidget(appearance_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_and_close)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _load_values(self):
        current_llm = self._cfg.get_str("llm/model")
        idx = self._llm_model_combo.findData(current_llm)
        if idx >= 0:
            self._llm_model_combo.setCurrentIndex(idx)

        self._whisper_model_input.setValue(
            int(self._cfg.get("model/whisper_model") == "tiny")
        )
        self._whisper_compute_input.setText(self._cfg.get_str("model/whisper_compute"))
        self._output_dir_input.setText(self._cfg.get_str("output/directory"))
        self._auto_save_input.setText(self._cfg.get_str("output/format"))
        self._font_size_input.setValue(self._cfg.get_int("appearance/font_size"))
        self._keywords_input.setText(self._cfg.get_str("appearance/keywords"))

    def _browse_output_dir(self):
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._output_dir_input.text()
        )
        if path:
            self._output_dir_input.setText(path)

    def _save_and_close(self):
        whisper_sizes = {0: "tiny", 1: "base", 2: "small", 3: "medium", 4: "large"}
        size = self._whisper_model_input.value()

        self._cfg.set("llm/model", self._llm_model_combo.currentData())
        self._cfg.set("model/whisper_model", whisper_sizes.get(size, "tiny"))
        self._cfg.set("model/whisper_compute", self._whisper_compute_input.text() or "int8")
        self._cfg.set("output/directory", self._output_dir_input.text())
        self._cfg.set("output/format", self._auto_save_input.text() or "txt")
        self._cfg.set("appearance/font_size", self._font_size_input.value())
        self._cfg.set("appearance/keywords", self._keywords_input.text())
        self.accept()
