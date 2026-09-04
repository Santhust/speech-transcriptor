from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter
from PySide6.QtWidgets import QWidget


class AudioMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self.setMinimumHeight(20)
        self.setMaximumHeight(30)
        self.setMinimumWidth(200)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)

    def set_level(self, level: float):
        self._level = max(0.0, min(1.0, level))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor(40, 40, 40))

        fill_w = int(w * self._level)

        if self._level < 0.6:
            color = QColor(50, 200, 50)
        elif self._level < 0.85:
            color = QColor(220, 180, 30)
        else:
            color = QColor(220, 50, 50)

        painter.fillRect(0, 0, fill_w, h, color)

        painter.setPen(QColor(80, 80, 80))
        painter.drawRect(0, 0, w - 1, h - 1)
        painter.end()
