import time
from pathlib import Path

import pytest

pytest.importorskip("torch")  # face recognition deps are optional; see requirements-recognition.txt

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from picsel import __version__  # noqa: E402


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


def test_metadata_loads_asynchronously_not_on_the_ui_thread(main_window, tmp_path, qapp):
    # Regression test: extract_metadata() does file I/O plus EXIF/GPS IFD
    # parsing, and previously ran synchronously on the UI thread on every
    # single photo navigation -- the one exception to this app's otherwise-
    # consistent "decode/detect off the UI thread" pattern (thumbnails,
    # full-image decode, and face detection are all already async).
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos)
    main_window.open_folder(photos)  # unlike library.load(), this actually triggers _show_current()

    # Right after _show_current(), a metadata worker should be queued but not
    # necessarily finished yet -- the panel should not have been populated
    # synchronously inline.
    assert main_window._pending_metadata_workers, "expected an async metadata worker to be queued"

    deadline = time.time() + 20
    while main_window._pending_metadata_workers and time.time() < deadline:
        time.sleep(0.02)
        qapp.processEvents()

    assert main_window.metadata_panel.tree.topLevelItemCount() > 0


def test_stale_metadata_result_is_discarded_after_rapid_navigation(main_window, tmp_path, qapp):
    # A metadata worker started for an earlier photo can finish after the
    # user has already navigated further -- its result must be dropped
    # rather than clobbering the panel with the wrong (now-stale) photo's
    # metadata.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=3)
    main_window.open_folder(photos)
    _drain_background_workers(main_window, qapp)
    while main_window._pending_metadata_workers:
        time.sleep(0.02)
        qapp.processEvents()

    current_snapshot = [
        main_window.metadata_panel.tree.topLevelItem(i).text(0)
        for i in range(main_window.metadata_panel.tree.topLevelItemCount())
    ]
    stale_generation = main_window._metadata_load_generation - 1
    assert stale_generation != main_window._metadata_load_generation

    fake_sections = [("Bogus Stale Section", [("field", "should never appear")])]
    main_window._on_metadata_loaded(stale_generation, photos / "img000.jpg", fake_sections, "", worker=None)

    after_snapshot = [
        main_window.metadata_panel.tree.topLevelItem(i).text(0)
        for i in range(main_window.metadata_panel.tree.topLevelItemCount())
    ]
    assert after_snapshot == current_snapshot, "a stale metadata result was not discarded"
    assert "Bogus Stale Section" not in after_snapshot


@pytest.mark.parametrize(
    "save_method, target, error_marker",
    [
        ("_save_library_state", "library", "save_state"),
        ("_save_face_catalog", "face_catalog", "save"),
        ("_save_person_gallery", "person_gallery", "save"),
    ],
)
def test_save_helpers_report_oserror_instead_of_raising(
    main_window, monkeypatch, save_method, target, error_marker
):
    # Regression test: every actual file operation elsewhere in main_window.py
    # (rename, save-as, culling) is wrapped in try/except OSError with a
    # message shown to the user -- the app's own state saves (ratings/status,
    # face cache, person gallery) previously weren't, so a folder on
    # removable/network media going unwritable mid-session could raise
    # straight out of an ordinary keyboard shortcut.
    def failing_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(getattr(main_window, target), error_marker, failing_save)

    captured = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: captured.append(a) or QMessageBox.StandardButton.Ok)
    )

    getattr(main_window, save_method)()  # must not raise

    assert captured, "expected a warning dialog instead of a raised OSError"


def test_face_filter_slider_is_debounced(main_window, tmp_path, qapp):
    # Regression test: the face-confidence threshold slider previously
    # rebuilt every visible face row and its full ranked person dropdown on
    # every single pixel of drag -- unlike the structurally similar edit
    # adjustment sliders, which are deliberately debounced via a QTimer to
    # avoid exactly this. _update_face_display() is wired up in __init__
    # (before this test can intercept it), so the debounce itself -- not
    # the downstream call -- is what's checked here: a burst of ticks must
    # coalesce into the timer firing once, single-shot, not redraw inline
    # on every tick.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos)
    main_window.open_folder(photos)
    main_window.side_tabs.setCurrentWidget(main_window.face_panel)
    _drain_background_workers(main_window, qapp)

    assert main_window._face_filter_timer.isSingleShot()
    assert not main_window._face_filter_timer.isActive()

    # Simulate a burst of slider ticks (one per pixel of drag) -- each one
    # should just (re)start the timer, not fire a redraw immediately.
    for value in range(10):
        main_window._on_face_filter_changed(value / 10.0)
        assert main_window._face_filter_timer.isActive()

    deadline = time.time() + 2
    while main_window._face_filter_timer.isActive() and time.time() < deadline:
        time.sleep(0.01)
        qapp.processEvents()

    assert not main_window._face_filter_timer.isActive()  # fired exactly once, single-shot


def test_window_title_includes_the_version(main_window):
    assert main_window.windowTitle() == f"picSel {__version__}"


def test_about_dialog_shows_the_version(main_window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "about", staticmethod(lambda parent, title, text: shown.append((title, text)))
    )

    main_window._show_about()

    assert shown
    title, text = shown[0]
    assert title == "About picSel"
    assert __version__ in text
