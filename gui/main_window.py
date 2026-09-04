import time

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPalette,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.audio_capture import AudioCapture
from core.engine_vosk import VoskEngine
from core.engine_whisper import WhisperEngine
from core.level_monitor import LevelMonitor
from core.output_manager import OutputManager
from core.summarizer import Summarizer
from gui.audio_meter import AudioMeter
from gui.search_bar import SearchBar, TranscriptHighlighter
from settings.config import LANGUAGE_MODELS, get_config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpeechTranscriptor")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)

        self._audio = AudioCapture()
        _cfg = get_config()
        self._language = _cfg.get_str("recognition/language")
        self._applied_theme = _cfg.get_str("appearance/theme")
        self._create_stt_engines()
        self._output = OutputManager()

        llm_model = _cfg.get_str("llm/model")
        self._summarizer = Summarizer(model_key=llm_model)

        self._transcript_segments: list[dict] = []
        self._is_streaming_mode = True
        self._batch_engine = "whisper"
        self._start_time = 0.0
        self._system_monitor = None
        self._mic_monitor = None
        self._vosk_mic: VoskEngine | None = None
        self._whisper_mic: WhisperEngine | None = None
        self._dual_mode: str | None = None
        self._partials: dict[str, str] = {}
        self._dual_pending: dict[str, list[dict]] = {}
        self._vosk_loaded = False
        self._whisper_loaded = False
        self._summarizer_loaded = False
        self._summarize_start_time = 0.0
        self._summarize_timer: QTimer | None = None
        self._last_partial_update = 0.0

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()
        self._apply_keywords_from_config()
        self._update_model_status()
        self.action_meter_system.setChecked(_cfg.get_bool("view/meter_system"))
        self.action_meter_mic.setChecked(_cfg.get_bool("view/meter_mic"))

        self._vosk.load_model()
        self._whisper.load_model()
        self._summarizer.load_model()

    def _create_stt_engines(self):
        lang = self._language
        _, vosk_model = LANGUAGE_MODELS.get(lang, LANGUAGE_MODELS["en"])
        whisper_lang = None if lang == "auto" else lang
        self._vosk = VoskEngine(model_name=vosk_model)
        self._whisper = WhisperEngine(language=whisper_lang)

    def _make_mic_engine_pair(self):
        _, vosk_model = LANGUAGE_MODELS.get(self._language, LANGUAGE_MODELS["en"])
        whisper_lang = None if self._language == "auto" else self._language
        self._vosk_mic = VoskEngine(model_name=vosk_model)
        self._whisper_mic = WhisperEngine(language=whisper_lang)

        self._vosk_mic.partial_result.connect(lambda t: self._set_partial("Mic", t))
        self._vosk_mic.final_result.connect(
            lambda text, ts: self._on_final(text, ts, speaker="Mic")
        )
        self._vosk_mic.transcription_ready.connect(
            lambda segs: self._stash_dual_result("Mic", segs)
        )
        self._vosk_mic.model_error.connect(self._on_model_error)

        self._whisper_mic.transcription_ready.connect(
            lambda segs: self._stash_dual_result("Mic", segs)
        )
        self._whisper_mic.transcription_error.connect(self._on_model_error)

    def _destroy_mic_engines(self):
        for attr in ("_vosk_mic", "_whisper_mic"):
            eng = getattr(self, attr, None)
            if eng is not None:
                eng.cleanup()
                eng.deleteLater()
                setattr(self, attr, None)

    def _rebuild_stt_engines(self):
        if hasattr(self, "_vosk"):
            self._vosk.cleanup()
            self._vosk.deleteLater()
        if hasattr(self, "_whisper"):
            self._whisper.cleanup()
            self._whisper.deleteLater()
        self._destroy_mic_engines()

        self._vosk_loaded = False
        self._whisper_loaded = False
        self._create_stt_engines()
        self._connect_engine_signals()
        self._update_model_status()

        display = LANGUAGE_MODELS.get(self._language, LANGUAGE_MODELS["en"])[0]
        self.statusBar().showMessage(
            f"Language changed to {display} — loading recognition models..."
        )
        self._vosk.load_model()
        self._whisper.load_model()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self._meters_panel = QWidget()
        meters_layout = QVBoxLayout(self._meters_panel)
        meters_layout.setContentsMargins(0, 0, 0, 0)
        meters_layout.setSpacing(2)

        self._system_meter_row = self._make_meter_row("speaker", "System")
        self._mic_meter_row = self._make_meter_row("mic", "Mic")
        meters_layout.addLayout(self._system_meter_row[0])
        meters_layout.addLayout(self._mic_meter_row[0])

        self._meters_panel.setVisible(False)
        layout.addWidget(self._meters_panel)

        self._search_bar = SearchBar()
        self._search_bar.search_changed.connect(self._on_search)
        layout.addWidget(self._search_bar)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText(
            "Select an audio device and press Ctrl+R to start recording..."
        )
        self.transcript_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        font = self.transcript_view.font()
        font.setPointSize(12)
        self.transcript_view.setFont(font)

        self._highlighter = TranscriptHighlighter(self.transcript_view)

        self._partial_header = QLabel("  Live Preview")
        self._partial_header.setFrameShape(QFrame.Shape.StyledPanel)
        self._partial_header.setStyleSheet(
            "QLabel { background-color: palette(base); font-weight: bold; "
            "font-size: 11px; color: palette(placeholderText); padding: 2px 6px; }"
        )

        self._partial_label = QTextEdit()
        self._partial_label.setReadOnly(True)
        self._partial_label.setPlaceholderText("Partial transcription will appear here...")
        pf = self._partial_label.font()
        pf.setPointSize(12)
        pf.setItalic(True)
        self._partial_label.setFont(pf)
        self._partial_label.setStyleSheet(
            "QTextEdit { color: palette(text); background-color: palette(base); }"
        )

        partial_container = QWidget()
        partial_layout = QVBoxLayout(partial_container)
        partial_layout.setContentsMargins(0, 0, 0, 0)
        partial_layout.setSpacing(0)
        partial_layout.addWidget(self._partial_header)
        partial_layout.addWidget(self._partial_label)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self.transcript_view)
        self._splitter.addWidget(partial_container)
        self._splitter.setStretchFactor(0, 7)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setSizes([500, 150])

        layout.addWidget(self._splitter)

        self._partial_label.setVisible(True)
        self._partial_header.setVisible(True)

    def _make_meter_row(self, kind: str, label_text: str):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        icon_label = QLabel()
        icon_label.setPixmap(self._source_icon(kind).pixmap(18, 18))
        icon_label.setFixedWidth(20)
        label = QLabel(label_text)
        label.setFixedWidth(64)
        meter = AudioMeter()
        meter.setVisible(False)
        row.addWidget(icon_label)
        row.addWidget(label)
        row.addWidget(meter)
        return row, meter, icon_label, kind

    @staticmethod
    def _icon_pen_color() -> QColor:
        color = QApplication.palette().color(QPalette.ColorRole.WindowText)
        if not color.isValid() or color.lightnessF() > 0.9:
            base = QApplication.palette().color(QPalette.ColorRole.Base)
            color = (
                QColor(Qt.white)
                if base.lightnessF() < 0.5
                else QColor(Qt.black)
            )
        return color

    @staticmethod
    def _make_icon(char: str, size: int = 24) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(MainWindow._icon_pen_color())
        painter.setFont(QFont("sans-serif", int(size * 0.65)))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, char)
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _make_device_icon(kind: str, size: int = 24) -> QIcon:
        from PySide6.QtCore import QPointF, QRectF

        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = size / 24.0
        color = MainWindow._icon_pen_color()

        def paint_speaker(p: QPainter):
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRect(QRectF(2 * s, 9 * s, 4 * s, 6 * s))
            p.drawPolygon(
                [
                    QPointF(6 * s, 9 * s),
                    QPointF(13 * s, 3 * s),
                    QPointF(13 * s, 21 * s),
                    QPointF(6 * s, 15 * s),
                ]
            )
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color, 1.8 * s))
            p.drawArc(QRectF(14 * s, 7 * s, 7 * s, 10 * s), -55 * 16, 110 * 16)
            p.drawArc(QRectF(17 * s, 4 * s, 13 * s, 16 * s), -55 * 16, 110 * 16)

        def paint_mic(p: QPainter):
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(9 * s, 2 * s, 6 * s, 11 * s), 3 * s, 3 * s)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color, 1.8 * s))
            p.drawArc(QRectF(5 * s, 6 * s, 14 * s, 13 * s), 180 * 16, -180 * 16)
            line_path = QPainterPath()
            line_path.moveTo(12 * s, 19 * s)
            line_path.lineTo(12 * s, 22 * s)
            p.drawPath(line_path)

        if kind == "speaker":
            paint_speaker(painter)
        elif kind == "mic":
            paint_mic(painter)
        elif kind == "mix":
            painter.save()
            painter.scale(0.62, 0.62)
            paint_speaker(painter)
            painter.restore()
            painter.save()
            painter.translate(11 * s, 3 * s)
            painter.scale(0.62, 0.62)
            paint_mic(painter)
            painter.restore()

        painter.end()
        return QIcon(pixmap)

    def _setup_toolbar(self):
        self._toolbar = QToolBar("Main Toolbar")
        self._toolbar.setMovable(False)
        self._toolbar.setIconSize(self._toolbar.iconSize())
        self.addToolBar(self._toolbar)

        self._device_combo = QComboBox()
        self._device_combo.setToolTip("Input device: system-audio monitor or microphone")
        self._device_combo.currentIndexChanged.connect(self._on_device_selected)
        self._refresh_device_combo()
        self._toolbar.addWidget(self._device_combo)

        self._apply_action_icons()

        self._toolbar.addSeparator()
        self._toolbar.addAction(self.action_start)
        self._toolbar.addAction(self.action_pause)
        self._toolbar.addAction(self.action_stop)

        self._toolbar.addSeparator()
        self._toolbar.addAction(self.action_streaming)
        self._toolbar.addAction(self.action_batch)

        self._toolbar.addSeparator()
        self._toolbar.addAction(self.action_export)
        self._toolbar.addAction(self.action_copy)

        self._toolbar.addSeparator()
        self._toolbar.addAction(self.action_summarize)
        self._toolbar.addAction(self.action_find)
        self._toolbar.addAction(self.action_clear)

    @staticmethod
    def _themed_icon(name: str, fallback: QIcon) -> QIcon:
        icon = QIcon.fromTheme(name)
        return icon if not icon.isNull() else fallback

    def _source_icon(self, kind: str) -> QIcon:
        if kind == "mic":
            return self._themed_icon(
                "audio-input-microphone", self._make_device_icon("mic")
            )
        if kind == "speaker":
            return self._themed_icon(
                "audio-volume-high", self._make_device_icon("speaker")
            )
        return self._make_device_icon("mix")

    def _apply_action_icons(self):
        self.action_start.setIcon(
            self._themed_icon("media-playback-start", self._make_icon("▶"))
        )
        self.action_pause.setIcon(
            self._themed_icon("media-playback-pause", self._make_icon("⏸"))
        )
        self.action_stop.setIcon(
            self._themed_icon("media-playback-stop", self._make_icon("⏹"))
        )
        self.action_streaming.setIcon(self._make_icon("📡"))
        self.action_batch.setIcon(self._make_icon("📄"))
        self.action_export.setIcon(
            self._themed_icon("document-save", self._make_icon("💾"))
        )
        self.action_copy.setIcon(self._themed_icon("edit-copy", self._make_icon("📋")))
        self.action_summarize.setIcon(self._make_icon("💡"))
        self.action_find.setIcon(self._themed_icon("edit-find", self._make_icon("🔍")))
        self.action_clear.setIcon(
            self._themed_icon("edit-clear", self._make_icon("\U0001f9f9"))
        )
        self._refresh_device_combo_icons()

    def _refresh_device_combo(self):
        cfg = get_config()
        saved_name = cfg.get_str("audio/device_name")
        devices = self._audio.get_all_devices()

        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._device_combo.addItem(
            "Mic + System (separate speakers)", "__dual_sep__"
        )
        self._device_combo.addItem(
            "Mic + System (merged)", "__dual_mix__"
        )
        self._refresh_device_combo_icons()
        for dev in devices:
            self._device_combo.addItem(dev.display_name, dev.pulse_source_name)

        idx = self._device_combo.findData(saved_name)
        if idx < 0 and saved_name not in ("__dual_sep__", "__dual_mix__"):
            default = self._audio.get_default_monitor()
            idx = (
                self._device_combo.findData(default.pulse_source_name)
                if default
                else 2
            )
        if idx < 0:
            idx = 0
        self._device_combo.setCurrentIndex(idx)
        cfg.set("audio/device_name", str(self._device_combo.itemData(idx)))
        self._device_combo.blockSignals(False)

    def _refresh_device_combo_icons(self):
        combo = getattr(self, "_device_combo", None)
        if combo is None or combo.count() == 0:
            return
        combo.setItemIcon(0, self._source_icon("mic"))
        combo.setItemIcon(1, self._make_device_icon("mix"))
        for i in range(2, combo.count()):
            source = combo.itemData(i)
            kind = (
                "speaker" if (source and str(source).endswith(".monitor")) else "mic"
            )
            combo.setItemIcon(i, self._source_icon(kind))

    def _on_device_selected(self, index: int):
        if index < 0:
            return
        source = self._device_combo.itemData(index)
        if source:
            get_config().set("audio/device_name", str(source))

    def _resolve_start_device(self):
        saved_name = get_config().get_str("audio/device_name")
        for dev in self._audio.get_all_devices():
            if dev.pulse_source_name == saved_name:
                return dev
        return self._audio.get_default_monitor()

    def _resolve_capture_plan(self):
        """Returns (mode, mic_dev, sys_dev, single_dev).

        mode: None for single-device recording, else "separate" or "merged".
        """
        saved_name = get_config().get_str("audio/device_name")
        if saved_name in ("__dual_sep__", "__dual_mix__"):
            mode = "separate" if saved_name == "__dual_sep__" else "merged"
            sys_dev = self._audio.get_default_monitor()
            mic_dev = next(
                (d for d in self._audio.get_all_devices() if not d.is_monitor), None
            )
            if sys_dev is not None and mic_dev is not None:
                return mode, mic_dev, sys_dev, None
        return None, None, None, self._resolve_start_device()

    def _meter_source(self, kind: str):
        if kind == "system":
            dev = self._audio.get_default_monitor()
            return dev.pulse_source_name if dev else None
        for dev in self._audio.get_all_devices():
            if not dev.is_monitor:
                return dev.pulse_source_name
        return None

    def _ensure_meter_monitor(self, kind: str):
        attr = f"_{kind}_monitor"
        monitor = getattr(self, attr)
        if monitor is not None:
            return monitor
        source = self._meter_source(kind)
        if source is None:
            return None
        meter = self._system_meter_row[1] if kind == "system" else self._mic_meter_row[1]
        monitor = LevelMonitor(source, lambda lvl: meter.set_level(lvl))
        setattr(self, attr, monitor)
        return monitor

    def _set_system_meter_visible(self, visible: bool):
        self._apply_meter_visibility("system", visible)

    def _set_mic_meter_visible(self, visible: bool):
        self._apply_meter_visibility("mic", visible)

    def _apply_meter_visibility(self, kind: str, visible: bool):
        cfg = get_config()
        cfg.set(f"view/meter_{kind}", bool(visible))
        row = self._system_meter_row if kind == "system" else self._mic_meter_row
        row[1].setVisible(visible)
        self._meters_panel.setVisible(
            self._system_meter_row[1].isVisibleTo(self._meters_panel)
            or self._mic_meter_row[1].isVisibleTo(self._meters_panel)
        )

        monitor = (
            self._ensure_meter_monitor(kind) if visible else getattr(self, f"_{kind}_monitor")
        )
        if monitor is not None:
            if visible:
                monitor.start()
                row[1].set_level(0.0)
            else:
                monitor.stop()

    def _toggle_all_meters(self):
        any_on = self.action_meter_system.isChecked() or self.action_meter_mic.isChecked()
        self.action_meter_system.setChecked(not any_on)
        self.action_meter_mic.setChecked(not any_on)

    def _shutdown_meter_monitors(self):
        for kind in ("system", "mic"):
            monitor = getattr(self, f"_{kind}_monitor", None)
            if monitor is not None:
                monitor.stop()

    def _setup_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        self.action_start = QAction("Start &Recording", self)
        self.action_start.setShortcut(QKeySequence("Ctrl+R"))
        self.action_start.triggered.connect(self._on_start)
        file_menu.addAction(self.action_start)

        self.action_stop = QAction("Stop &Recording", self)
        self.action_stop.setShortcut(QKeySequence("Ctrl+S"))
        self.action_stop.setEnabled(False)
        self.action_stop.triggered.connect(self._on_stop)
        file_menu.addAction(self.action_stop)

        self.action_pause = QAction("&Pause/Resume", self)
        self.action_pause.setShortcut(QKeySequence("Ctrl+P"))
        self.action_pause.setEnabled(False)
        self.action_pause.triggered.connect(self._on_pause)
        file_menu.addAction(self.action_pause)

        file_menu.addSeparator()

        self.action_export = QAction("&Export Last Transcript", self)
        self.action_export.setShortcut(QKeySequence("Ctrl+E"))
        self.action_export.triggered.connect(self._on_export)
        file_menu.addAction(self.action_export)

        self.action_open_folder = QAction("&Open Output Folder", self)
        self.action_open_folder.setShortcut(QKeySequence("Ctrl+O"))
        self.action_open_folder.triggered.connect(self._on_open_folder)
        file_menu.addAction(self.action_open_folder)

        file_menu.addSeparator()

        self.action_summarize = QAction("&Summarize Transcript", self)
        self.action_summarize.setShortcut(QKeySequence("Ctrl+U"))
        self.action_summarize.setEnabled(False)
        self.action_summarize.triggered.connect(self._on_summarize)
        file_menu.addAction(self.action_summarize)

        file_menu.addSeparator()

        action_quit = QAction("&Quit", self)
        action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        action_quit.triggered.connect(self.close)
        file_menu.addAction(action_quit)

        edit_menu = menubar.addMenu("&Edit")

        self.action_copy = QAction("&Copy Transcript", self)
        self.action_copy.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.action_copy.triggered.connect(self._copy_transcript)
        edit_menu.addAction(self.action_copy)

        action_clear = QAction("&Clear Display", self)
        action_clear.setIcon(self._make_icon("\U0001f9f9"))
        action_clear.setShortcut(QKeySequence("Ctrl+L"))
        action_clear.triggered.connect(self._clear_display)
        self.action_clear = action_clear
        edit_menu.addAction(action_clear)

        edit_menu.addSeparator()

        self.action_find = QAction("&Find...", self)
        self.action_find.setShortcut(QKeySequence("Ctrl+F"))
        self.action_find.triggered.connect(self._on_find)
        edit_menu.addAction(self.action_find)

        edit_menu.addSeparator()

        action_prefs = QAction("&Preferences...", self)
        action_prefs.setShortcut(QKeySequence("Ctrl+,"))
        action_prefs.triggered.connect(self._on_preferences)
        edit_menu.addAction(action_prefs)

        view_menu = menubar.addMenu("&View")

        self.action_streaming = QAction("Streaming Mode", self)
        self.action_streaming.setShortcut(QKeySequence("Ctrl+1"))
        self.action_streaming.setCheckable(True)
        self.action_streaming.setChecked(True)
        self.action_streaming.triggered.connect(lambda: self._set_mode(True))
        view_menu.addAction(self.action_streaming)

        self.action_batch = QAction("Batch Mode", self)
        self.action_batch.setShortcut(QKeySequence("Ctrl+2"))
        self.action_batch.setCheckable(True)
        self.action_batch.triggered.connect(lambda: self._set_mode(False))
        view_menu.addAction(self.action_batch)

        view_menu.addSeparator()

        batch_engine_menu = view_menu.addMenu("Batch &Engine")

        self.action_batch_vosk = QAction("Vosk", self)
        self.action_batch_vosk.setCheckable(True)
        self.action_batch_vosk.triggered.connect(lambda: self._set_batch_engine("vosk"))
        batch_engine_menu.addAction(self.action_batch_vosk)

        self.action_batch_whisper = QAction("faster-whisper", self)
        self.action_batch_whisper.setCheckable(True)
        self.action_batch_whisper.setChecked(True)
        self.action_batch_whisper.triggered.connect(lambda: self._set_batch_engine("whisper"))
        batch_engine_menu.addAction(self.action_batch_whisper)

        view_menu.addSeparator()

        meters_menu = view_menu.addMenu("Audio &Meters")

        self.action_meter_system = QAction("Show \U0001f50a System Meter", self)
        self.action_meter_system.setCheckable(True)
        self.action_meter_system.toggled.connect(self._set_system_meter_visible)
        meters_menu.addAction(self.action_meter_system)

        self.action_meter_mic = QAction("Show \U0001f3a4 Microphone Meter", self)
        self.action_meter_mic.setCheckable(True)
        self.action_meter_mic.toggled.connect(self._set_mic_meter_visible)
        meters_menu.addAction(self.action_meter_mic)

        self.action_meter = QAction("Toggle All &Meters", self)
        self.action_meter.setShortcut(QKeySequence("Ctrl+M"))
        self.action_meter.triggered.connect(self._toggle_all_meters)
        meters_menu.addAction(self.action_meter)

        view_menu.addSeparator()
        view_menu.addAction(self.action_clear)

        self.action_autoscroll = QAction("&Auto-scroll", self)
        self.action_autoscroll.setShortcut(QKeySequence("Ctrl+J"))
        self.action_autoscroll.setCheckable(True)
        self.action_autoscroll.setChecked(True)
        view_menu.addAction(self.action_autoscroll)

        self.action_timestamps = QAction("Show &Timestamps", self)
        self.action_timestamps.setShortcut(QKeySequence("Ctrl+T"))
        self.action_timestamps.setCheckable(True)
        self.action_timestamps.setChecked(True)
        view_menu.addAction(self.action_timestamps)

        help_menu = menubar.addMenu("&Help")

        action_doc = QAction("&Documentation", self)
        action_doc.setShortcut(QKeySequence("F1"))
        action_doc.triggered.connect(self._on_documentation)
        help_menu.addAction(action_doc)

        action_dev_test = QAction("&Audio Device Test", self)
        action_dev_test.triggered.connect(self._on_device_test)
        help_menu.addAction(action_dev_test)

        action_engine_test = QAction("&Test STT Engine", self)
        action_engine_test.triggered.connect(self._on_engine_test)
        help_menu.addAction(action_engine_test)

        action_llm_test = QAction("Test &LLM Summarization", self)
        action_llm_test.triggered.connect(self._on_llm_test)
        help_menu.addAction(action_llm_test)

        help_menu.addSeparator()

        self.action_retry_models = QAction("&Download/Retry Models", self)
        self.action_retry_models.triggered.connect(self._on_retry_models)
        help_menu.addAction(self.action_retry_models)

        help_menu.addSeparator()

        action_about = QAction("&About SpeechTranscriptor", self)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)

    def _setup_statusbar(self):
        self._status_model_label = QLabel("")
        self.statusBar().addPermanentWidget(self._status_model_label)

        self._download_label = QLabel("")
        self._download_label.setVisible(False)
        self._download_bar = QProgressBar()
        self._download_bar.setFixedWidth(180)
        self._download_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._download_label)
        self.statusBar().addPermanentWidget(self._download_bar)

        self._last_download_emit = 0.0
        self.statusBar().showMessage("Initializing models...")

    def _on_download_progress(self, done: int, total: int):
        if total <= 0:
            self._hide_download_progress()
            return

        now = time.time()
        if done < total and (now - self._last_download_emit) < 0.1:
            return
        self._last_download_emit = now

        pct = min(int(done * 100 / total), 100)
        name = "model"
        if isinstance(self.sender(), VoskEngine):
            name = self.sender()._model_name
        elif isinstance(self.sender(), WhisperEngine):
            name = f"faster-whisper-{self.sender()._model_size}"
        self._download_label.setText(f"Downloading {name}:")
        self._download_bar.setValue(pct)
        mb_done = done / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        self._download_bar.setFormat(f"{pct}% ({mb_done:.0f}/{mb_total:.0f} MB)")
        self._download_label.setVisible(True)
        self._download_bar.setVisible(True)

    def _hide_download_progress(self):
        self._download_label.setVisible(False)
        self._download_bar.setVisible(False)

    def _update_model_status(self):
        parts = []
        parts.append("Vosk: " + ("ready" if self._vosk_loaded else "loading..."))
        parts.append("Whisper: " + ("ready" if self._whisper_loaded else "loading..."))
        parts.append("LLM: " + ("ready" if self._summarizer_loaded else "loading..."))
        self._status_model_label.setText("  |  ".join(parts))

    def _set_mode(self, streaming: bool):
        self._is_streaming_mode = streaming
        self.action_streaming.setChecked(streaming)
        self.action_batch.setChecked(not streaming)

        self._partial_label.setVisible(streaming)
        self._partial_header.setVisible(streaming)

        if streaming:
            self._splitter.setSizes([500, 150])
        else:
            self._splitter.setSizes([700, 0])

    def _set_batch_engine(self, engine: str):
        self._batch_engine = engine
        self.action_batch_vosk.setChecked(engine == "vosk")
        self.action_batch_whisper.setChecked(engine == "whisper")

    def _connect_signals(self):
        self._audio.set_audio_callback(self._on_audio_data)

        self._connect_engine_signals()

        self._summarizer.model_loading.connect(
            lambda: self.statusBar().showMessage("Loading summarization model...")
        )
        self._summarizer.model_loaded.connect(self._on_summarizer_loaded)
        self._summarizer.summary_ready.connect(self._on_summary_ready)
        self._summarizer.summary_error.connect(self._on_model_error)

    def _connect_engine_signals(self):
        self._vosk.partial_result.connect(lambda t: self._set_partial("System", t))
        self._vosk.final_result.connect(self._on_final)
        self._vosk.transcription_ready.connect(self._on_batch_result)
        self._vosk.model_loaded.connect(self._on_vosk_loaded)
        self._vosk.model_error.connect(self._on_model_error)
        self._vosk.download_progress.connect(self._on_download_progress)

        self._whisper.transcription_ready.connect(self._on_batch_result)
        self._whisper.transcription_error.connect(self._on_model_error)
        self._whisper.model_loaded.connect(self._on_whisper_loaded)
        self._whisper.download_progress.connect(self._on_download_progress)

    @Slot(str)
    def _on_vosk_loaded(self, name: str):
        self._vosk_loaded = True
        self._update_model_status()
        self._check_all_loaded()

    @Slot(str)
    def _on_whisper_loaded(self, name: str):
        self._whisper_loaded = True
        self._update_model_status()
        self._check_all_loaded()
    def _check_all_loaded(self):
        if self._vosk_loaded and self._whisper_loaded and self._summarizer_loaded:
            self.statusBar().showMessage("All models ready — press Ctrl+R to record")
        elif self._vosk_loaded and self._whisper_loaded:
            self.statusBar().showMessage("Vosk + Whisper ready — LLM still loading...")
        else:
            loaded = []
            if self._vosk_loaded:
                loaded.append("Vosk")
            if self._whisper_loaded:
                loaded.append("Whisper")
            self.statusBar().showMessage(f"Loaded: {', '.join(loaded)} — waiting for others...")

    @Slot(str)
    def _on_model_error(self, error: str):
        self._hide_download_progress()
        if self._summarize_timer and self._summarize_timer.isActive():
            elapsed = time.time() - self._summarize_start_time
            self._summarize_timer.stop()
            self.action_summarize.setEnabled(True)
            self.statusBar().showMessage(f"Summarizer error ({elapsed:.1f}s): {error}", 5000)
        else:
            self.statusBar().showMessage(f"Model error: {error}")

    def _on_audio_data(self, audio_data, sample_rate: int, source_idx: int = -1):
        if source_idx == 1:
            if self._is_streaming_mode:
                if self._vosk_mic is not None:
                    self._vosk_mic.feed_audio(audio_data, sample_rate)
            elif self._batch_engine == "vosk":
                if self._vosk_mic is not None:
                    self._vosk_mic.feed_audio_buffered(audio_data, sample_rate)
            else:
                if self._whisper_mic is not None:
                    self._whisper_mic.feed_audio(audio_data, sample_rate)
            return

        if self._is_streaming_mode:
            self._vosk.feed_audio(audio_data, sample_rate)
        else:
            if self._batch_engine == "vosk":
                self._vosk.feed_audio_buffered(audio_data, sample_rate)
            else:
                self._whisper.feed_audio(audio_data, sample_rate)

    def _set_partial(self, speaker: str, text: str):
        now = time.time()
        if not self._is_streaming_mode or (now - self._last_partial_update) < 0.15:
            return
        self._last_partial_update = now

        if self._dual_mode == "separate":
            icon = "\U0001f3a4" if speaker == "Mic" else "\U0001f50a"
            self._partials[speaker] = f"{icon} {text}"
            other = "System" if speaker == "Mic" else "Mic"
            self._partials.setdefault(other, "")
            lines = [
                self._partials[k]
                for k in ("Mic", "System")
                if self._partials.get(k)
            ]
            self._partial_label.setPlainText("\n".join(lines))
        else:
            self._partial_label.setPlainText(text)

    @Slot(str, float)
    def _on_final(self, text: str, timestamp: float, speaker: str | None = None):
        self._partials.pop(speaker or "System", None)
        if self._dual_mode == "separate" and not any(self._partials.values()):
            self._partials.clear()
            self._partial_label.clear()
        elapsed = timestamp - self._start_time if self._start_time else 0
        self._add_segment(text, elapsed, speaker)

    @Slot(list)
    def _on_batch_result(self, segments: list[dict]):
        if self._dual_mode == "separate":
            self._stash_dual_result("System", segments)
            return
        self._render_batch(segments)

    def _stash_dual_result(self, speaker: str, segments: list[dict]):
        self._dual_pending[speaker] = list(segments)
        if "System" in self._dual_pending and "Mic" in self._dual_pending:
            merged = []
            for spk, segs in self._dual_pending.items():
                for seg in segs:
                    merged.append({**seg, "speaker": spk})
            merged.sort(key=lambda s: s.get("start", 0))
            self._dual_pending.clear()
            self._render_batch(merged)

    def _render_batch(self, segments: list[dict]):
        self._partial_label.clear()
        self._transcript_segments.clear()
        self.transcript_view.clear()

        for seg in segments:
            self._add_segment(seg["text"], seg.get("start", 0), seg.get("speaker"))

        engine = "Vosk" if self._batch_engine == "vosk" else "Whisper"
        self.statusBar().showMessage(
            f"Batch transcription complete — {len(segments)} segments ({engine})"
        )

    def _add_segment(self, text: str, start_time: float, speaker: str | None = None):
        if speaker:
            text = f"[{speaker}] {text}"
        entry = {"text": text, "start": start_time}
        if speaker:
            entry["speaker"] = speaker
        self._transcript_segments.append(entry)

        if self.action_timestamps.isChecked():
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            self.transcript_view.append(f"[{minutes:02d}:{seconds:02d}] {text}")
        else:
            self.transcript_view.append(text)

        if self.action_autoscroll.isChecked():
            QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        sb = self.transcript_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_start(self):
        mode, mic_dev, sys_dev, single_dev = self._resolve_capture_plan()
        device = single_dev
        if mode is None and device is None:
            self.statusBar().showMessage("No audio input device found!")
            return

        if mode == "separate":
            self._dual_mode = "separate"
            if self._vosk_mic is None and self._whisper_mic is None:
                self._make_mic_engine_pair()
            mic_vosk_needed = (
                self._is_streaming_mode or self._batch_engine == "vosk"
            )
            if mic_vosk_needed and not self._vosk_mic.is_loaded():
                self._vosk_mic.load_model()
                self.statusBar().showMessage(
                    "Loading second Vosk model for microphone — press Record again in a moment..."
                )
                return
            if not mic_vosk_needed and not self._whisper_mic.is_loaded():
                self._whisper_mic.load_model()
                self.statusBar().showMessage(
                    "Loading second Whisper model for microphone — press Record again in a moment..."
                )
                return
        else:
            self._dual_mode = None

        if self._is_streaming_mode and not self._vosk.is_loaded():
            self.statusBar().showMessage("Vosk model still loading, please wait...")
            return
        if not self._is_streaming_mode:
            if self._batch_engine == "vosk" and not self._vosk.is_loaded():
                self.statusBar().showMessage("Vosk model still loading, please wait...")
                return
            if self._batch_engine == "whisper" and not self._whisper.is_loaded():
                self.statusBar().showMessage("Whisper model still loading, please wait...")
                return

        self._dual_pending.clear()
        self._partials.clear()
        self._vosk.reset()
        self._vosk.clear_buffer()
        self._whisper.clear_buffer()
        if self._vosk_mic is not None:
            self._vosk_mic.reset()
            self._vosk_mic.clear_buffer()
        if self._whisper_mic is not None:
            self._whisper_mic.clear_buffer()
        self._transcript_segments.clear()
        self._start_time = time.time()

        if mode is not None:
            self._audio.start_dual(mic_dev, sys_dev, mode)
        else:
            self._audio.select_device(device)
            self._audio.start(device)

        self.action_start.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.action_pause.setEnabled(True)
        self.action_pause.setText("&Pause")

        if self._is_streaming_mode:
            engine_label = "Streaming (Vosk)"
        elif self._batch_engine == "vosk":
            engine_label = "Batch (Vosk)"
        else:
            engine_label = "Batch (Whisper)"

        if mode == "separate":
            label = f"{engine_label} [Mic + System, separate]"
        elif mode == "merged":
            label = f"{engine_label} [Mic + System, merged]"
        else:
            label = f"{engine_label}: {device.display_name}"
        self.statusBar().showMessage(f"Recording [{label}]")

    def _on_stop(self):
        self._audio.stop()

        dual_sep = self._dual_mode == "separate"

        if self._is_streaming_mode:
            final_text = self._vosk.get_final()
            if final_text:
                self._on_final(final_text, time.time())
            if dual_sep and self._vosk_mic is not None:
                mic_text = self._vosk_mic.get_final()
                if mic_text:
                    self._on_final(mic_text, time.time(), speaker="Mic")
            self.statusBar().showMessage(
                f"Stopped — {len(self._transcript_segments)} segments transcribed"
            )
        else:
            if dual_sep:
                self.statusBar().showMessage(
                    "Stopped — transcribing both sources (mic + system)..."
                )
            else:
                self.statusBar().showMessage("Stopped — transcribing...")

            if self._batch_engine == "vosk":
                duration = self._vosk.get_buffer_duration()
                self.statusBar().showMessage(
                    f"Stopped — {duration:.1f}s captured, transcribing with Vosk..."
                )
                self._vosk.transcribe_batch()
                if dual_sep and self._vosk_mic is not None:
                    self._vosk_mic.transcribe_batch()
            else:
                duration = self._whisper.get_buffer_duration()
                self.statusBar().showMessage(
                    f"Stopped — {duration:.1f}s captured, transcribing with Whisper..."
                )
                self._whisper.transcribe()
                if dual_sep and self._whisper_mic is not None:
                    self._whisper_mic.transcribe()

        self._dual_mode = None
        self.action_start.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.action_pause.setEnabled(False)
        self.action_pause.setText("&Pause")

    def _on_pause(self):
        if self.action_pause.text() == "&Pause":
            self._audio.pause()
            self.action_pause.setText("&Resume")
            self.statusBar().showMessage("Paused")
        else:
            self._audio.resume()
            self.action_pause.setText("&Pause")
            self.statusBar().showMessage("Recording...")

    def _on_export(self):
        if not self._transcript_segments:
            self.statusBar().showMessage("No transcript to export", 3000)
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Transcript",
            str(self._output.get_output_dir() / "transcript.txt"),
            "Text Files (*.txt);;SRT Files (*.srt);;All Files (*)",
        )
        if path:
            if path.endswith(".srt"):
                self._output.save_srt(self._transcript_segments, filename=path)
            else:
                self._output.save_txt(self._transcript_segments, filename=path)
            self.statusBar().showMessage(f"Exported to {path}", 3000)

    def _on_open_folder(self):
        import subprocess
        subprocess.Popen(["xdg-open", str(self._output.get_output_dir())])

    def _copy_transcript(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.transcript_view.toPlainText())
        self.statusBar().showMessage("Transcript copied to clipboard", 3000)

    def _clear_display(self):
        self.transcript_view.clear()
        self._partial_label.clear()
        self._transcript_segments.clear()
        self._partials.clear()
        self._dual_pending.clear()

    def _on_summarize(self):
        text = self.transcript_view.toPlainText()
        if not text.strip() and self._transcript_segments:
            text = "\n".join(
                f"[{s['start']:.0f}s] {s['text']}" for s in self._transcript_segments
            )
        if not text.strip():
            self.statusBar().showMessage("No transcript to summarize", 3000)
            return
        self.action_summarize.setEnabled(False)
        self._summarize_start_time = time.time()
        self._summarize_timer = QTimer(self)
        self._summarize_timer.timeout.connect(self._update_summarize_progress)
        self._summarize_timer.start(500)
        self.statusBar().showMessage(
            f"Generating summary with {self._summarizer.current_model_display()}..."
        )
        self._summarizer.summarize(text)

    def _update_summarize_progress(self):
        elapsed = time.time() - self._summarize_start_time
        self.statusBar().showMessage(f"Generating summary... ({elapsed:.0f}s elapsed)")

    @Slot()
    def _on_summarizer_loaded(self):
        self._summarizer_loaded = True
        self.action_summarize.setEnabled(True)
        self._update_model_status()
        self._check_all_loaded()

    @Slot(str)
    def _on_summary_ready(self, summary: str):
        elapsed = time.time() - self._summarize_start_time
        if self._summarize_timer:
            self._summarize_timer.stop()
        self.action_summarize.setEnabled(True)
        self.transcript_view.append("\n" + "=" * 40 + "\n")
        self.transcript_view.append("SUMMARY:\n")
        self.transcript_view.append(summary)
        self.statusBar().showMessage(f"Summary generated in {elapsed:.1f}s", 5000)
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _on_preferences(self):
        from gui.settings_dialog import SettingsDialog

        dlg = SettingsDialog(self)
        if dlg.exec():
            from settings.config import get_config
            cfg = get_config()
            keywords = cfg.get_str("appearance/keywords")
            kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
            self._highlighter.set_keywords(kw_list)

            new_model = cfg.get_str("llm/model")
            if new_model != self._summarizer.current_model_key():
                self._summarizer.switch_model(new_model)

            new_lang = cfg.get_str("recognition/language")
            if new_lang != self._language:
                self._language = new_lang
                self._rebuild_stt_engines()

            new_theme = cfg.get_str("appearance/theme")
            if new_theme != self._applied_theme:
                self._switch_theme(new_theme)

    def _switch_theme(self, theme: str):
        from settings.theme import apply_theme

        apply_theme(theme)
        self._applied_theme = theme
        self._apply_action_icons()
        for row in (self._system_meter_row, self._mic_meter_row):
            row[2].setPixmap(self._source_icon(row[3]).pixmap(18, 18))
            self._repolish(row[2])
            self._repolish(row[1])
        for widget in (self._partial_header, self._partial_label):
            self._repolish(widget)
            viewport = getattr(widget, "viewport", None)
            if callable(viewport):
                self._repolish(viewport())

    @staticmethod
    def _repolish(widget):
        if widget is None:
            return
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _on_find(self):
        self._search_bar.toggle()

    def _on_search(self, term: str):
        self._highlighter.set_search_term(term)

    def _apply_keywords_from_config(self):
        from settings.config import get_config
        cfg = get_config()
        keywords = cfg.get_str("appearance/keywords")
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        self._highlighter.set_keywords(kw_list)

    def _on_documentation(self):
        from PySide6.QtWidgets import QDialog, QLabel, QTextBrowser

        dlg = QDialog(self)
        dlg.setWindowTitle("Documentation")
        dlg.setMinimumSize(550, 480)

        layout = QVBoxLayout(dlg)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(
            "<h2>SpeechTranscriptor — Help</h2>"
            "<h3>Quick Start</h3>"
            "<ol>"
            "<li>Play audio in any app (browser, music player, video call)</li>"
            "<li>Press <b>Ctrl+R</b> (or the toolbar Record button) to start capturing</li>"
            "<li>Speak or let audio play — transcription appears live in the bottom panel</li>"
            "<li>Press <b>Ctrl+S</b> (or toolbar Stop) to finish</li>"
            "</ol>"

            "<h3>Modes</h3>"
            "<p><b>Streaming Mode</b> (default): Real-time transcription using Vosk. "
            "Partial text appears live as audio is captured. Best for live events.</p>"
            "<p><b>Batch Mode</b>: Captures all audio first, then transcribes at once "
            "when you press Stop. Supports both Vosk and faster-whisper engines. "
            "Better accuracy for longer recordings.</p>"

            "<h3>Keyboard Shortcuts</h3>"
            "<table cellpadding='4'>"
            "<tr><td><b>Ctrl+R</b></td><td>Start recording</td></tr>"
            "<tr><td><b>Ctrl+S</b></td><td>Stop recording</td></tr>"
            "<tr><td><b>Ctrl+P</b></td><td>Pause / Resume</td></tr>"
            "<tr><td><b>Ctrl+E</b></td><td>Export transcript</td></tr>"
            "<tr><td><b>Ctrl+O</b></td><td>Open output folder</td></tr>"
            "<tr><td><b>Ctrl+U</b></td><td>Summarize transcript (LLM)</td></tr>"
            "<tr><td><b>Ctrl+1 / Ctrl+2</b></td><td>Switch Streaming / Batch mode</td></tr>"
            "<tr><td><b>Ctrl+F</b></td><td>Find in transcript</td></tr>"
            "<tr><td><b>Ctrl+T</b></td><td>Toggle timestamps</td></tr>"
            "<tr><td><b>Ctrl+J</b></td><td>Toggle auto-scroll</td></tr>"
            "<tr><td><b>Ctrl+M</b></td><td>Show audio level meter</td></tr>"
            "<tr><td><b>Ctrl+L</b></td><td>Clear display</td></tr>"
            "<tr><td><b>Ctrl+Shift+C</b></td><td>Copy transcript to clipboard</td></tr>"
            "<tr><td><b>Ctrl+,</b></td><td>Preferences</td></tr>"
            "<tr><td><b>F1</b></td><td>This help page</td></tr>"
            "</table>"

            "<h3>Toolbar</h3>"
            "<p>The toolbar provides quick access to all major actions with icon + text labels. "
            "Recording controls, mode switching, export, copy, summarize, and search.</p>"

            "<h3>Export</h3>"
            "<p>Exports can be saved as plain text (.txt) or subtitles (.srt) format. "
            "Use the Preferences dialog to set your default output directory and format.</p>"

            "<h3>Models</h3>"
            "<p>First run will download models automatically (~40 MB for Vosk, ~40 MB for Whisper, "
            "~2 GB for LLM summarization). Check Help > Download/Retry Models if a download failed.</p>"
        )
        layout.addWidget(browser)

        from PySide6.QtWidgets import QPushButton
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec()

    def _on_retry_models(self):
        self.statusBar().showMessage("Retrying model downloads...")
        if not self._vosk_loaded:
            self._vosk.load_model()
        if not self._whisper_loaded:
            self._whisper.load_model()
        if not self._summarizer_loaded:
            self._summarizer.load_model()

    def _on_device_test(self):
        from gui.diagnostics_dialog import DeviceTestDialog
        dlg = DeviceTestDialog(self)
        dlg.exec()

    def _on_engine_test(self):
        from gui.diagnostics_dialog import EngineTestDialog
        dlg = EngineTestDialog(self)
        dlg.exec()

    def _on_llm_test(self):
        from gui.llm_test_dialog import LlmTestDialog
        dlg = LlmTestDialog(self)
        dlg.exec()

    def _on_about(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About SpeechTranscriptor",
            "<h2>SpeechTranscriptor</h2>"
            "<p>Desktop application for real-time system audio transcription.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Streaming transcription (Vosk)</li>"
            "<li>Batch transcription (Vosk or faster-whisper)</li>"
            "<li>Local LLM summarization (Qwen2.5-3B)</li>"
            "<li>Export to .txt and .srt</li>"
            "<li>Search and keyword highlighting</li>"
            "</ul>"
            "<p>Press Ctrl+R to start recording.</p>"
            "<p>Licensed under GPL-3.0 — see LICENSE.</p>",
        )

    def closeEvent(self, event):
        self._audio.stop()
        self._shutdown_meter_monitors()
        self._destroy_mic_engines()
        self._vosk.cleanup()
        self._whisper.cleanup()
        self._summarizer.cleanup()
        event.accept()
