import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication


THEME_OPTIONS = {
    "system": "System",
    "light": "Light",
    "dark": "Dark",
}

_icon_pair: tuple[str, str] | None = None  # (light_theme, dark_theme)


def apply_theme(theme: str):
    global _icon_pair
    app = QApplication.instance()
    if app is None:
        return

    from PySide6.QtGui import QIcon

    _ensure_search_paths()
    if _icon_pair is None:
        _icon_pair = _resolve_icon_pair()

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


def _is_dark_theme_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith("-dark") or lowered.endswith("_dark") or "dark" in lowered


def _light_counterpart(name: str) -> str:
    for suffix in ("-dark", "_dark"):
        if name.lower().endswith(suffix):
            stripped = name[: -len(suffix)]
            if _theme_installed(stripped):
                return stripped
    for candidate in ("breeze", "Adwaita", "Tango", "PiXtrix", "oxygen"):
        if _theme_installed(candidate) and not _is_dark_theme_name(candidate):
            return candidate
    return name


def _dark_counterpart(name: str) -> str:
    if _is_dark_theme_name(name):
        return name
    for candidate in (f"{name}-dark", f"{name}_dark"):
        if _theme_installed(candidate):
            return candidate
    return name


def _resolve_icon_pair() -> tuple[str, str]:
    from PySide6.QtGui import QIcon

    base = QIcon.themeName() or _detect_fallback_theme()
    light = _light_counterpart(base) if base else _detect_fallback_theme()
    dark = _dark_counterpart(base if base else light)
    return light, dark


def _sync_icon_theme(theme: str):
    if _icon_pair is None:
        return
    light, dark = _icon_pair

    target: str | None = None
    if theme == "dark":
        target = dark if _theme_installed(dark) else None
    else:
        target = light if _theme_installed(light) else None

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
