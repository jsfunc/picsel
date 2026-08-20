import logging
import sys

import pytest
from PySide6.QtWidgets import QMessageBox

import tamis.app as app_module


@pytest.fixture(autouse=True)
def _reset_logging():
    # logging.basicConfig() is a global, process-wide side effect --
    # isolate each test from whatever the previous one configured, and
    # restore sys.excepthook so a test installing the real hook doesn't
    # leak into pytest's own error reporting for later tests.
    original_excepthook = sys.excepthook
    yield
    sys.excepthook = original_excepthook
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_configure_logging_creates_a_persistent_log_file(tmp_path, monkeypatch):
    log_path = tmp_path / ".tamis" / "tamis.log"
    monkeypatch.setattr(app_module, "LOG_PATH", log_path)

    app_module._configure_logging()
    logging.getLogger("tamis.somewhere").warning("a warning that must not vanish")

    assert log_path.exists()
    assert "a warning that must not vanish" in log_path.read_text()


def test_exception_hook_logs_and_shows_a_dialog_instead_of_crashing_silently(tmp_path, monkeypatch):
    log_path = tmp_path / ".tamis" / "tamis.log"
    monkeypatch.setattr(app_module, "LOG_PATH", log_path)
    app_module._configure_logging()
    app_module._install_exception_hook()

    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *a, **k: shown.append(a) or QMessageBox.StandardButton.Ok),
    )

    try:
        raise ValueError("something went wrong deep in a slot")
    except ValueError:
        sys.excepthook(*sys.exc_info())  # simulates what Qt/Python would call for an uncaught exception

    assert shown, "expected a QMessageBox.critical dialog to be shown"
    log_text = log_path.read_text()
    assert "Unhandled exception" in log_text
    assert "ValueError: something went wrong deep in a slot" in log_text


def test_exception_hook_does_not_intercept_keyboard_interrupt(tmp_path, monkeypatch):
    log_path = tmp_path / ".tamis" / "tamis.log"
    monkeypatch.setattr(app_module, "LOG_PATH", log_path)
    app_module._configure_logging()
    app_module._install_exception_hook()

    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *a, **k: shown.append(a) or QMessageBox.StandardButton.Ok),
    )

    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        sys.excepthook(*sys.exc_info())

    assert not shown  # Ctrl+C shouldn't pop an "unexpected error" dialog
