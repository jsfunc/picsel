"""Application bootstrap for picSel."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QLoggingCategory
from PySide6.QtWidgets import QApplication

from picsel.main_window import MainWindow

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def run(initial_folder: str | None = None) -> int:
    # Silences benign "OpenType support missing for <font>, script N" notices:
    # Qt logs these whenever a system font lacks shaping tables for some
    # script it probed while building its font database. It doesn't affect
    # rendering (Qt just falls back to a font that does support the script).
    QLoggingCategory.setFilterRules("qt.text.font.db=false")

    app = QApplication(sys.argv)
    app.setApplicationName("picSel")

    window = MainWindow()
    window.show()

    if initial_folder:
        window.open_folder(Path(initial_folder))

    return app.exec()
