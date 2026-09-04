from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


THEME_OPTIONS = {
    "system": "System",
    "light": "Light",
    "dark": "Dark",
}


def apply_theme(theme: str):
    app = QApplication.instance()
    if app is None:
        return

    if theme == "dark":
        app.setStyle("Fusion")
        app.setPalette(_dark_palette())
    elif theme == "light":
        app.setStyle("Fusion")
        app.setPalette(app.style().standardPalette())
    else:
        app.setPalette(app.style().standardPalette())


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.WindowText, Qt.white)
    p.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ToolTipBase, Qt.white)
    p.setColor(QPalette.ColorRole.ToolTipText, Qt.white)
    p.setColor(QPalette.ColorRole.Text, Qt.white)
    p.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    p.setColor(QPalette.ColorRole.ButtonText, Qt.white)
    p.setColor(QPalette.ColorRole.BrightText, Qt.red)
    p.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, Qt.black)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    p.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127)
    )
    return p
