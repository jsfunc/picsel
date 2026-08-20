import time
from pathlib import Path

import pytest

pytest.importorskip("torch")  # face recognition deps are optional; see requirements-recognition.txt

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QDialog, QMessageBox  # noqa: E402

import tamis.controllers.face_recognition_controller as face_ctl_module  # noqa: E402
import tamis.main_window as mw_module  # noqa: E402
from tamis import __version__  # noqa: E402
from tamis.models import Status  # noqa: E402


def _make_photos(folder: Path, count: int = 2) -> None:
    for i in range(count):
        Image.new("RGB", (40, 30), (10 * i, 20, 30)).save(folder / f"img{i:03d}.jpg")


def _drain_background_workers(main_window, qapp, timeout: float = 20.0) -> None:
    # QTest.qWait() doesn't reliably release the GIL in this PySide6 binding
    # (observed elsewhere this session starving a QThreadPool worker for 90+
    # seconds) -- poll with a plain sleep + processEvents instead.
    deadline = time.time() + timeout
    while (
        main_window._pending_image_workers or main_window.face_ctl._pending_face_workers
    ) and time.time() < deadline:
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
    "get_target, get_save_method, error_marker",
    [
        (lambda mw: mw.library, lambda mw: mw._save_library_state, "save_state"),
        # face_catalog/person_gallery saves run on a background thread now (see
        # FaceRecognitionController._save_thread_pool) -- write_payload is the
        # part that actually touches disk, so that's what needs to fail here,
        # not save() (which now only does the cheap, synchronous snapshot).
        (lambda mw: mw.face_ctl.face_catalog, lambda mw: mw.face_ctl.save_face_catalog, "write_payload"),
        (lambda mw: mw.face_ctl.person_gallery, lambda mw: mw.face_ctl.save_person_gallery, "write_payload"),
    ],
)
def test_save_helpers_report_oserror_instead_of_raising(
    main_window, tmp_path, monkeypatch, qapp, get_target, get_save_method, error_marker
):
    # Regression test: every actual file operation elsewhere in main_window.py
    # (rename, save-as, culling) is wrapped in try/except OSError with a
    # message shown to the user -- the app's own state saves (ratings/status,
    # face cache, person gallery) previously weren't, so a folder on
    # removable/network media going unwritable mid-session could raise
    # straight out of an ordinary keyboard shortcut.

    # face_catalog.prepare_save() short-circuits (no write at all) unless a
    # folder is loaded with at least one cached record -- give it that, so
    # write_payload actually gets reached for that row's monkeypatch below
    # (harmless setup for the other two rows).
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos)
    main_window.open_folder(photos)
    main_window.face_ctl.face_catalog.add_manual_face(main_window.library.items[0].path, box=(1, 1, 10, 10))

    def failing_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(get_target(main_window), error_marker, failing_save)

    captured = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: captured.append(a) or QMessageBox.StandardButton.Ok)
    )

    get_save_method(main_window)()  # must not raise

    # The face_catalog/person_gallery cases dispatch to a background thread
    # and the warning arrives via a queued cross-thread signal -- give the
    # event loop a chance to deliver it (a no-op wait for the synchronous
    # library-save case, since captured is already non-empty by then).
    deadline = time.time() + 5
    while not captured and time.time() < deadline:
        time.sleep(0.02)
        qapp.processEvents()

    assert captured, "expected a warning dialog instead of a raised OSError"


def test_face_catalog_and_gallery_saves_do_not_block_the_ui_thread(main_window, tmp_path, qapp):
    # Regression test: face_catalog.save()/person_gallery.save() used to
    # rewrite their entire file synchronously on the UI thread on every
    # single call -- confirming one face name felt laggy (measured ~0.6s for
    # a moderately-used folder/gallery) because the click blocked on disk
    # I/O + JSON serialization. save_face_catalog()/save_person_gallery()
    # now only do a cheap synchronous snapshot and defer the slow part to a
    # background thread -- verified here by making the write itself
    # artificially slow and confirming the calling method still returns
    # almost immediately, then confirming the data actually lands on disk
    # once wait_for_pending_saves() returns.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos)
    main_window.open_folder(photos)
    main_window.face_ctl.face_catalog.add_manual_face(main_window.library.items[0].path, box=(1, 1, 10, 10))
    main_window.face_ctl.person_gallery.add_person("Alice")

    real_write_faces = main_window.face_ctl.face_catalog.write_payload
    real_write_gallery = main_window.face_ctl.person_gallery.write_payload

    def slow_write(write_fn):
        def wrapped(path, data):
            time.sleep(0.3)
            write_fn(path, data)

        return wrapped

    main_window.face_ctl.face_catalog.write_payload = slow_write(real_write_faces)
    main_window.face_ctl.person_gallery.write_payload = slow_write(real_write_gallery)

    t0 = time.time()
    main_window.face_ctl.save_face_catalog()
    main_window.face_ctl.save_person_gallery()
    elapsed = time.time() - t0

    assert elapsed < 0.1, f"save_face_catalog()/save_person_gallery() blocked for {elapsed:.2f}s -- should return immediately"

    main_window.face_ctl.wait_for_pending_saves()

    reloaded_catalog_json = main_window.face_ctl.face_catalog._state_path().read_text()
    assert "person_id" in reloaded_catalog_json  # face_catalog.save()'s format landed on disk

    import gzip
    import json

    reloaded_gallery = json.loads(gzip.decompress(main_window.face_ctl.person_gallery.path.read_bytes()))
    assert any(p["name"] == "Alice" for p in reloaded_gallery["people"])


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

    assert main_window.face_ctl._face_filter_timer.isSingleShot()
    assert not main_window.face_ctl._face_filter_timer.isActive()

    # Simulate a burst of slider ticks (one per pixel of drag) -- each one
    # should just (re)start the timer, not fire a redraw immediately.
    for value in range(10):
        main_window.face_ctl._on_face_filter_changed(value / 10.0)
        assert main_window.face_ctl._face_filter_timer.isActive()

    deadline = time.time() + 2
    while main_window.face_ctl._face_filter_timer.isActive() and time.time() < deadline:
        time.sleep(0.01)
        qapp.processEvents()

    assert not main_window.face_ctl._face_filter_timer.isActive()  # fired exactly once, single-shot


def test_window_title_includes_the_version(main_window):
    assert main_window.windowTitle() == f"Tamis {__version__}"


def test_about_dialog_shows_the_version(main_window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "about", staticmethod(lambda parent, title, text: shown.append((title, text)))
    )

    main_window._show_about()

    assert shown
    title, text = shown[0]
    assert title == "About Tamis"
    assert __version__ in text


def _auto_accept_apply_culling(monkeypatch) -> None:
    monkeypatch.setattr(mw_module.ApplyCullingDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))


def test_apply_culling_move_does_not_leave_a_stale_rating_for_the_moved_photo(
    main_window, tmp_path, qapp, monkeypatch
):
    # Regression test: save_state() previously ran *before* the post-move
    # reload, while self.library.items still listed the just-moved photo --
    # writing its rating into the parent folder's .tamis_state.json even
    # though the photo no longer lives there. If a future photo reused that
    # exact filename in this folder, it would silently inherit the stale
    # rating/status.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=2)
    main_window.open_folder(photos)
    main_window.library.set_status(0, Status.SELECTED)
    main_window.library.set_rating(0, 5)
    moved_name = main_window.library.items[0].name

    _auto_accept_apply_culling(monkeypatch)
    main_window._apply_culling()

    state_path = photos / ".tamis_state.json"
    if state_path.exists():
        assert moved_name not in state_path.read_text()
    assert (photos / "selected" / moved_name).exists()


def test_apply_culling_move_invalidates_cached_face_data_for_the_moved_photo(
    main_window, tmp_path, qapp, monkeypatch
):
    # Regression test: same bug as the ratings/status one above, but for
    # cached face detections -- face_catalog.save() ran before the reload
    # too, and FaceCatalog.load() (unlike ImageLibrary.load()) never
    # cross-checks its entries against what's actually still on disk, so
    # reordering alone wouldn't have fixed this half; the moved photo's
    # cached record must be explicitly invalidated before the save.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=2)
    main_window.open_folder(photos)
    moved_item = main_window.library.items[0]
    moved_name = moved_item.name
    main_window.face_ctl.face_catalog.add_manual_face(moved_item.path, box=(5, 5, 20, 20))
    main_window.library.set_status(0, Status.SELECTED)

    _auto_accept_apply_culling(monkeypatch)
    main_window._apply_culling()

    faces_path = photos / ".tamis_faces.json"
    if faces_path.exists():
        assert moved_name not in faces_path.read_text()
    assert moved_name not in main_window.face_ctl.face_catalog._records


def test_face_detection_gets_the_same_elevated_priority_as_image_loads(main_window, tmp_path, qapp, monkeypatch):
    # Regression test: the visible full-image load is deliberately given
    # elevated thread-pool priority over background thumbnail decoding, so
    # the photo you're looking at doesn't wait behind a large folder's
    # thumbnail queue -- face detection didn't get the same treatment, so
    # switching to the Face Recognition tab right after opening a large
    # folder could queue detection for the current photo behind thousands
    # of pending thumbnail jobs.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos)
    main_window.open_folder(photos)

    calls = []
    real_start = main_window._thread_pool.start

    def spying_start(worker, priority=0):
        calls.append((worker, priority))
        return real_start(worker, priority)

    monkeypatch.setattr(main_window._thread_pool, "start", spying_start)

    main_window.side_tabs.setCurrentWidget(main_window.face_panel)
    _drain_background_workers(main_window, qapp)

    face_calls = [call for call in calls if isinstance(call[0], face_ctl_module.FaceDetectionWorker)]
    assert face_calls, "no FaceDetectionWorker was started"
    assert all(priority == mw_module.IMAGE_LOAD_PRIORITY for _, priority in face_calls)


def test_bundled_resource_path_resolves_relative_to_the_repo_when_not_frozen():
    path = mw_module._bundled_resource_path("docs/face_recognition.html")
    assert path.name == "face_recognition.html"
    assert path.exists()  # the real docs file, resolved from source (not frozen in tests)


def test_open_face_recognition_docs_opens_the_real_bundled_file(main_window, monkeypatch):
    opened = []
    monkeypatch.setattr(mw_module.QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url)))

    main_window._open_face_recognition_docs()

    assert opened
    assert opened[0].toLocalFile().endswith("docs/face_recognition.html")
    assert Path(opened[0].toLocalFile()).exists()


def test_open_face_recognition_docs_warns_instead_of_opening_a_missing_file(main_window, monkeypatch):
    monkeypatch.setattr(
        mw_module, "_bundled_resource_path", lambda relative: Path("/nonexistent/face_recognition.html")
    )
    opened = []
    monkeypatch.setattr(mw_module.QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url)))
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a) or QMessageBox.StandardButton.Ok)
    )

    main_window._open_face_recognition_docs()

    assert not opened
    assert warned
