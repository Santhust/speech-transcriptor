import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication


THEME_OPTIONS = {
    "system": "System",
    "light": "Light",
    "dark": "Dark",
}

_original_icon_theme: str | None = None


def apply_theme(theme: str):
    global _original_icon_theme
    app = QApplication.instance()
    if app is None:
        return

    from PySide6.QtGui import QIcon

    _ensure_search_paths()
    if _original_icon_theme is None:
        _original_icon_theme = QIcon.themeName() or _detect_fallback_theme()

    if theme == "dark":
        app.setStyle("Fusion")
        app.setPalette(_dark_palette())
    elif theme == "light":
        app.setStyle("Fusion")
        app.setPalette(app.style().standardPalette())
    else:
        app.setPalette(app.style().standardPalette())

    _sync_icon_theme(theme)


def _ensure_search_paths():
    paths = QIcon.themeSearchPaths()
    changed = False
    for candidate in (
        "/usr/share/icons",
        "/usr/local/share/icons",
        os.path.expanduser("~/.local/share/icons"),
    ):
        if candidate not in paths and os.path.isdir(candidate):
            paths.append(candidate)
            changed = True
    if changed:
        QIcon.setThemeSearchPaths(paths)


def _theme_installed(name: str | None) -> bool:
    if not name:
        return False
    for search_path in QIcon.themeSearchPaths():
        if search_path.startswith(":"):
            continue
        path = search_path.replace("$HOME", os.path.expanduser("~"))
        if os.path.exists(os.path.join(path, name, "index.theme")):
            return True
    return False


def _detect_fallback_theme() -> str:
    for name in ("breeze", "Adwaita", "Tango", "PiXtrix", "oxygen", "hicolor"):
        if _theme_installed(name):
            return name
    return ""


def _sync_icon_theme(theme: str):
    base = _original_icon_theme or ""

    target: str | None = None
    if theme == "dark":
        for candidate in (f"{base}-dark", f"{base}_dark", "breeze-dark", "Adwaita-dark"):
            if _theme_installed(candidate):
                target = candidate
                break
        if target is None:
            target = base if _theme_installed(base) else None
    else:
        if base and _theme_installed(base):
            target = base
        else:
            detected = _detect_fallback_theme()
            target = detected or None

    if target:
        QIcon.setThemeName(target)


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
