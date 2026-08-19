"""Application bootstrap for picSel."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from PySide6.QtCore import QLoggingCategory
from PySide6.QtWidgets import QApplication, QMessageBox

from picsel import __version__
from picsel.main_window import MainWindow

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

LOG_PATH = Path.home() / ".picsel" / "picsel.log"


def _configure_logging() -> None:
    """Give every `logging.getLogger(...)` call in the app somewhere
    persistent to go. Without this, nothing was ever configured (Python's
    logging module does nothing until something calls basicConfig or adds a
    handler), so any warning/error logged anywhere -- e.g. thumbnails.py's
    own logger -- went nowhere durable. That matters most for the packaged
    executable, where most users have no visible console to see stray
    output on even if there were any.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    # force=True: basicConfig() is documented to do nothing if the root
    # logger already has handlers (e.g. a test runner's own log-capture
    # plugin) -- without it, this call could silently fail to install the
    # file handler depending on what else happened to run first.
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def _install_exception_hook() -> None:
    """Log (and tell the user about) any exception that escapes all the way
    up the call stack, instead of the previous behavior for anything not
    explicitly caught inline: the app just vanishing or freezing with zero
    explanation and no trace of what happened.
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)  # still visible if run from a terminal
        try:
            QMessageBox.critical(
                None,
                "Unexpected Error",
                "picSel hit an unexpected error and may be in an inconsistent state -- "
                "saving your work and restarting is recommended.\n\n"
                f"{exc_type.__name__}: {exc_value}\n\n"
                f"Details were written to {LOG_PATH}",
            )
        except Exception:
            pass  # a failure showing the dialog shouldn't mask the original exception

    sys.excepthook = handle_exception


def run(initial_folder: str | None = None) -> int:
    _configure_logging()
    _install_exception_hook()

    # Silences benign "OpenType support missing for <font>, script N" notices:
    # Qt logs these whenever a system font lacks shaping tables for some
    # script it probed while building its font database. It doesn't affect
    # rendering (Qt just falls back to a font that does support the script).
    QLoggingCategory.setFilterRules("qt.text.font.db=false")

    app = QApplication(sys.argv)
    app.setApplicationName("picSel")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    window.show()

    if initial_folder:
        window.open_folder(Path(initial_folder))

    return app.exec()
