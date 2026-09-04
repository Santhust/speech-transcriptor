import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from settings.config import get_config
from settings.theme import apply_theme


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpeechTranscriptor")
    app.setOrganizationName("SpeechTranscriptor")

    apply_theme(get_config().get_str("appearance/theme"))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
