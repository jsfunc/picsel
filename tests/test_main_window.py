import time
from pathlib import Path

import pytest

pytest.importorskip("torch")  # face recognition deps are optional; see requirements-recognition.txt

from PIL import Image  # noqa: E402


def _make_photos(folder: Path, count: int = 2) -> None:
    for i in range(count):
        Image.new("RGB", (40, 30), (10 * i, 20, 30)).save(folder / f"img{i:03d}.jpg")


def _drain_background_workers(main_window, qapp, timeout: float = 20.0) -> None:
    # QTest.qWait() doesn't reliably release the GIL in this PySide6 binding
    # (observed elsewhere this session starving a QThreadPool worker for 90+
    # seconds) -- poll with a plain sleep + processEvents instead.
    deadline = time.time() + timeout
    while (main_window._pending_image_workers or main_window._pending_face_workers) and time.time() < deadline:
        time.sleep(0.05)
        qapp.processEvents()


def test_switching_tabs_turns_off_crop_and_face_edit_modes(main_window, tmp_path, qapp):
    # Regression test for a real bug: crop mode and face-edit mode only
    # defended against *each other*, not against a tab switch -- the image
    # viewer isn't itself part of any tab, so it stays visible and
    # interactive no matter which side tab is active. Leaving either mode on
    # while switching tabs let a drag on the photo silently crop it, or
    # silently add a face box the user never meant to draw.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos)
    main_window.library.load(photos)
    qapp.processEvents()

    # Turn on Edit Faces, then switch away to the Edit Image tab.
    main_window.side_tabs.setCurrentWidget(main_window.face_panel)
    qapp.processEvents()
    main_window.face_panel.edit_button.setChecked(True)
    qapp.processEvents()
    assert main_window.viewer._face_edit_mode is True

    main_window.side_tabs.setCurrentWidget(main_window.edit_panel)
    qapp.processEvents()
    assert main_window.viewer._face_edit_mode is False
    assert main_window.face_panel.edit_button.isChecked() is False

    # Turn on Crop, then switch away to the Face Recognition tab.
    main_window.edit_panel.crop_button.setChecked(True)
    qapp.processEvents()
    assert main_window.viewer._crop_mode is True

    main_window.side_tabs.setCurrentWidget(main_window.face_panel)
    qapp.processEvents()
    assert main_window.viewer._crop_mode is False
    assert main_window.edit_panel.crop_button.isChecked() is False


def test_face_edit_mode_does_not_carry_over_to_the_next_photo(main_window, tmp_path, qapp):
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos)
    main_window.library.load(photos)
    qapp.processEvents()

    main_window.side_tabs.setCurrentWidget(main_window.face_panel)
    qapp.processEvents()
    main_window.face_panel.edit_button.setChecked(True)
    qapp.processEvents()
    assert main_window.viewer._face_edit_mode is True

    main_window._go_next()
    _drain_background_workers(main_window, qapp)

    assert main_window.viewer._face_edit_mode is False
    assert main_window.face_panel.edit_button.isChecked() is False
