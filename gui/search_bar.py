from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QTextCharFormat
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SearchBar(QWidget):
    search_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 2)

        row.addWidget(QLabel("Find:"))
        self._input = QLineEdit()
        self._input.setPlaceholderText("Search transcript...")
        self._input.returnPressed.connect(self._on_search)
        self._input.textChanged.connect(self._on_search)
        row.addWidget(self._input)

        self._count_label = QLabel("")
        row.addWidget(self._count_label)

        close_btn = QPushButton("×")
        close_btn.setFixedWidth(24)
        close_btn.clicked.connect(self.hide)
        close_btn.setShortcut(QKeySequence("Escape"))
        row.addWidget(close_btn)

        layout.addLayout(row)

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self._input.setFocus()
            self._input.selectAll()

    def _on_search(self):
        text = self._input.text()
        self.search_changed.emit(text)


class TranscriptHighlighter:
    def __init__(self, text_edit: QTextEdit):
        self._text_edit = text_edit
        self._highlight_format = QTextCharFormat()
        self._highlight_format.setBackground(QColor(255, 255, 0, 180))
        self._search_format = QTextCharFormat()
        self._search_format.setBackground(QColor(0, 120, 215, 150))
        self._search_format.setForeground(QColor(255, 255, 255))
        self._keywords: list[str] = []
        self._search_term: str = ""

    def set_keywords(self, keywords: list[str]):
        self._keywords = [k.lower() for k in keywords if k.strip()]
        self.refresh()

    def set_search_term(self, term: str):
        self._search_term = term.lower()
        self.refresh()

    def refresh(self):
        doc = self._text_edit.document()
        cursor = self._text_edit.textCursor()
        cursor.beginEditBlock()

        for block_idx in range(doc.blockCount()):
            block = doc.findBlockByNumber(block_idx)
            block_text = block.text().lower()

            start = block.position()
            for char_idx in range(len(block.text())):
                for kw in self._keywords:
                    if block_text[char_idx:].startswith(kw):
                        cursor.setPosition(start + char_idx)
                        cursor.movePosition(
                            cursor.MoveOperation.Right,
                            cursor.MoveMode.KeepAnchor,
                            len(kw),
                        )
                        cursor.setCharFormat(self._highlight_format)

                if self._search_term and block_text[char_idx:].startswith(self._search_term):
                    term_len = len(self._search_term)
                    cursor.setPosition(start + char_idx)
                    cursor.movePosition(
                        cursor.MoveOperation.Right,
                        cursor.MoveMode.KeepAnchor,
                        term_len,
                    )
                    cursor.setCharFormat(self._search_format)

        cursor.endEditBlock()

    def highlight_matches(self, term: str):
        self._text_edit.setExtraSelections([])
        if not term:
            return

        extra_selections = []
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(0, 120, 215, 150))
        fmt.setForeground(QColor(255, 255, 255))

        cursor = self._text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)

        while True:
            cursor = self._text_edit.document().find(term, cursor)
            if cursor.isNull():
                break
            sel = QTextEdit.ExtraSelection()
            sel.format = fmt
            sel.cursor = cursor
            extra_selections.append(sel)

        self._text_edit.setExtraSelections(extra_selections)
