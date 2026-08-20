import os
import sys
from pathlib import Path

# Must be set before any PySide6 import happens (anywhere, including inside
# tamis modules imported by test files below) -- there's no display in a
# test/CI environment, and the offscreen platform plugin needs to be chosen
# before QApplication is ever touched.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for the whole test session -- Qt only allows one
    per process, and constructing/tearing one down per test is unnecessary
    overhead for what's currently just a handful of MainWindow-level tests."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_window(qapp, tmp_path):
    """A real MainWindow, safe to construct in tests: PersonGallery's default
    path is redirected to an isolated tmp_path location first, so no test
    can ever read from or overwrite the real ~/.tamis/people.json.gz --
    MainWindow.__init__ takes no constructor parameters, so this is the only
    way to make it safe (see the "no injectable paths" finding from the
    architecture review). PersonGallery itself is only imported inside
    FaceRecognitionController now (main_window.py delegates face-recognition
    state to it), so that's what needs patching, not main_window directly.
    """
    import tamis.controllers.face_recognition_controller as face_ctl_module
    import tamis.main_window as mw_module

    original_defaults = face_ctl_module.PersonGallery.__init__.__defaults__
    face_ctl_module.PersonGallery.__init__.__defaults__ = (tmp_path / "people.json.gz",)
    try:
        window = mw_module.MainWindow()
        yield window
    finally:
        window.close()
        face_ctl_module.PersonGallery.__init__.__defaults__ = original_defaults
