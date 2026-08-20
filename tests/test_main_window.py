import json
import time
from collections import deque
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
        main_window._pending_image_workers
        or main_window.face_ctl._pending_face_workers
        # Detection is debounced, so it may not have been enqueued yet -- a
        # drain that ignored the pending timer would return before any
        # FaceDetectionWorker existed to wait for.
        or main_window.face_ctl._detection_timer.isActive()
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


def test_face_detection_never_runs_on_the_shared_thread_pool(main_window, tmp_path, qapp, monkeypatch):
    # Detection used to run on the shared pool with elevated priority, so the
    # visible photo wouldn't queue behind thumbnail work. Priority orders the
    # queue but cannot preempt a running task, and detection blocks on the
    # model lock while holding its thread -- so browsing past enough photos
    # filled every shared thread with detection workers waiting on each other,
    # and decode for the photo actually on screen had nowhere to run
    # (measured: 152ms -> 2103ms). It now has a pool of its own, which is a
    # stronger guarantee than any priority: it cannot occupy a shared thread
    # at all.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos)
    main_window.open_folder(photos)

    started_on_shared = []
    real_start = main_window._thread_pool.start

    def spying_start(worker, priority=0):
        started_on_shared.append(worker)
        return real_start(worker, priority)

    monkeypatch.setattr(main_window._thread_pool, "start", spying_start)

    main_window.side_tabs.setCurrentWidget(main_window.face_panel)
    _drain_background_workers(main_window, qapp)

    assert not [w for w in started_on_shared if isinstance(w, face_ctl_module.FaceDetectionWorker)]
    assert main_window.face_ctl._detection_pool.maxThreadCount() == 1


def test_the_photo_being_viewed_outranks_speculative_warming(main_window, tmp_path, qapp, monkeypatch):
    # The detection pool runs one job at a time, so priority strictly orders
    # its queue: whatever the user is looking at must be ahead of every
    # already-navigated-past photo queued behind it.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=4)
    main_window.open_folder(photos)

    started = []
    monkeypatch.setattr(
        main_window.face_ctl._detection_pool, "start", lambda w, p=0: started.append((w.path, p))
    )
    main_window.face_ctl._active = True
    for _ in range(3):
        main_window._go_next()
    main_window.face_ctl._start_pending_detection()

    assert started, "no detection was enqueued"
    current = main_window.library.current_item.path
    assert started[0][0] == current
    assert started[0][1] == face_ctl_module._CURRENT_PHOTO_PRIORITY
    assert all(priority == face_ctl_module._WARM_PRIORITY for _, priority in started[1:])
    assert all(path != current for path, _ in started[1:])


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


def _stub_face(main_window, folder: Path, filename: str, embedding):
    """Put one face record in the catalog for `filename` and make it the
    controller's current photo, without running the detector."""
    from tamis.recognition.faces import FaceRecord

    ctl = main_window.face_ctl
    path = folder / filename
    record = FaceRecord(box=(0, 0, 9, 9), confidence=0.99, embedding=embedding)
    ctl.face_catalog.folder = folder
    ctl.face_catalog._records[filename] = [record]

    class _Item:
        def __init__(self, p):
            self.path = p
            self.name = p.name

    main_window.library.items = [_Item(path)]
    main_window.library.current_index = 0
    ctl._current_face_path = path
    ctl._current_face_records = [record]
    ctl._update_face_display()
    return record


def test_confirming_a_name_after_a_merge_in_another_folder_moves_the_sample(main_window, tmp_path):
    """A merge done while a folder was closed used to leave that folder's face
    records pointing at a deleted person id. The labeling path reads an
    unresolvable id as "not labeled yet", so re-confirming the name added a
    *second* copy of the sample while the orphaned first copy stayed put --
    the same face then voted for two people, and `identify` broke the tie
    arbitrarily. Measured on a real gallery: 34% of samples were duplicates.
    """
    import numpy as np

    ctl = main_window.face_ctl
    gallery = ctl.person_gallery
    folder_a = tmp_path / "A"
    folder_a.mkdir()
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)

    record = _stub_face(main_window, folder_a, "a.jpg", embedding)
    ctl._on_face_name_confirmed(0, "Arnaud")
    merged_away = gallery.find_by_name("Arnaud")
    assert len(merged_away.embeddings) == 1

    # Someone merges 'Arnaud' into 'Arnaud Fauchon' with a different folder open.
    ctl.face_catalog.load(tmp_path / "B")
    keep = gallery.add_person("Arnaud Fauchon")
    gallery.add_embedding(keep.id, np.random.default_rng(1).normal(size=512).astype(np.float32))
    gallery.merge(keep_id=keep.id, remove_id=merged_away.id)
    samples_after_merge = len(keep.embeddings)

    # Reopen folder A: its record still names the merged-away person.
    ctl.face_catalog.load(folder_a)
    ctl.face_catalog._records["a.jpg"] = [record]
    ctl.face_catalog.reconcile_people(gallery.merged_ids, {p.id for p in gallery.people})
    _stub_face(main_window, folder_a, "a.jpg", embedding)
    ctl._current_face_records = [record]
    ctl._current_visible_face_records = [record]

    ctl._on_face_name_confirmed(0, "Arnaud Fauchon")

    assert len(gallery.find_by_name("Arnaud Fauchon").embeddings) == samples_after_merge
    assert sum(
        1 for s in gallery.find_by_name("Arnaud Fauchon").embeddings if np.allclose(s, embedding, atol=1e-2)
    ) == 1
    assert record.person_id == keep.id


def test_confirming_a_name_on_a_record_naming_a_forgotten_person_does_not_duplicate(main_window, tmp_path):
    """Same failure mode, reached via Forget Name rather than a merge -- there
    is no successor id to redirect to, so the stale label is cleared instead.
    """
    import numpy as np

    ctl = main_window.face_ctl
    gallery = ctl.person_gallery
    folder = tmp_path / "A"
    folder.mkdir()
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)

    record = _stub_face(main_window, folder, "a.jpg", embedding)
    record.person_id = "a-person-who-no-longer-exists"
    ctl._current_visible_face_records = [record]

    ctl._on_face_name_confirmed(0, "Alice")

    alice = gallery.find_by_name("Alice")
    assert len(alice.embeddings) == 1
    assert record.person_id == alice.id


def test_an_unreadable_gallery_does_not_wipe_a_folders_face_labels(main_window, tmp_path, monkeypatch):
    """A gallery that fails to read loads empty, which would make every label
    in every folder afterwards look like it names a person who no longer
    exists. Reconciling against that would clear them all and save the result,
    turning a failed read into permanent data loss.
    """
    import numpy as np

    from tamis.recognition.faces import FACES_FILENAME, FaceCatalog, FaceRecord

    ctl = main_window.face_ctl
    folder = tmp_path / "photos"
    folder.mkdir()

    person = ctl.person_gallery.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    ctl.person_gallery.add_embedding(person.id, embedding)

    catalog = FaceCatalog()
    catalog.load(folder)
    record = FaceRecord(box=(0, 0, 9, 9), confidence=0.99, embedding=embedding)
    record.person_id = person.id
    catalog._records["a.jpg"] = [record]
    catalog.save()

    # Now simulate the gallery file having become unreadable.
    ctl.person_gallery.people = []
    ctl.person_gallery.load_error = "could not read the gallery"
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    ctl.load_folder(folder)

    assert ctl.face_catalog._records["a.jpg"][0].person_id == person.id
    on_disk = json.loads((folder / FACES_FILENAME).read_text())
    assert on_disk["a.jpg"][0]["person_id"] == person.id


def test_fast_navigation_enqueues_no_detection_for_photos_passed_through(main_window, tmp_path, monkeypatch):
    # The core fix: arrowing enqueued one full detection per photo, all
    # serialized by the model lock, so the photo finally landed on completed
    # last -- 3.3s for the 12th of 12 on a fast GPU.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=6)
    main_window.open_folder(photos)

    started = []
    monkeypatch.setattr(
        main_window.face_ctl._detection_pool, "start", lambda w, p=0: started.append((w.path, p))
    )
    main_window.face_ctl._active = True
    for _ in range(5):
        main_window._go_next()  # faster than the debounce; nothing should start yet

    assert started == []
    assert main_window.face_ctl._detection_timer.isActive()


def test_a_photo_navigated_past_is_queued_to_warm_the_cache(main_window, tmp_path, monkeypatch):
    # Opportunistic: a skipped photo's detection result is cached per-photo,
    # so the work stays useful if the user comes back to it. It just must not
    # outrank the photo being looked at.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=3)
    main_window.open_folder(photos)

    monkeypatch.setattr(main_window.face_ctl._detection_pool, "start", lambda w, p=0: None)
    ctl = main_window.face_ctl
    ctl._active = True
    first = main_window.library.items[0].path
    ctl.request_detection()
    main_window._go_next()
    ctl.request_detection()

    assert first in ctl._warm_queue


def test_an_already_cached_photo_is_not_queued_for_warming(main_window, tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=3)
    main_window.open_folder(photos)

    monkeypatch.setattr(main_window.face_ctl._detection_pool, "start", lambda w, p=0: None)
    ctl = main_window.face_ctl
    ctl._active = True
    first = main_window.library.items[0].path
    ctl.face_catalog._records[first.name] = []  # already detected
    ctl.request_detection()
    main_window._go_next()
    ctl.request_detection()

    assert first not in ctl._warm_queue


def test_the_warm_queue_is_bounded(main_window, tmp_path, monkeypatch):
    # An unbounded queue would let a long browse commit the GPU to minutes of
    # work for photos the user has already left behind.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=3)
    main_window.open_folder(photos)
    monkeypatch.setattr(main_window.face_ctl._detection_pool, "start", lambda w, p=0: None)

    ctl = main_window.face_ctl
    for i in range(face_ctl_module._MAX_WARM_QUEUE * 3):
        ctl._warm_queue.append(tmp_path / f"far{i}.jpg")

    assert len(ctl._warm_queue) == face_ctl_module._MAX_WARM_QUEUE


def test_switching_folders_abandons_queued_detection(main_window, tmp_path, monkeypatch):
    # Results for the old folder are discarded anyway (faces_for refuses to
    # cache across a folder switch), so running them would only occupy the GPU
    # while the new folder's photos wait.
    first = tmp_path / "first"
    first.mkdir()
    _make_photos(first, count=3)
    second = tmp_path / "second"
    second.mkdir()
    _make_photos(second, count=1)
    main_window.open_folder(first)

    monkeypatch.setattr(main_window.face_ctl._detection_pool, "start", lambda w, p=0: None)
    ctl = main_window.face_ctl
    ctl._active = True
    ctl.request_detection()
    ctl._warm_queue.append(first / "img000.jpg")
    ctl._start_pending_detection()
    queued = list(ctl._pending_face_workers)
    assert queued, "expected queued detection work"

    main_window.open_folder(second)

    assert all(w.cancelled for w in queued)
    assert ctl._warm_queue == deque()
    assert ctl._queued_paths == set()


def test_a_cancelled_worker_does_not_run_detection(tmp_path):
    from tamis.recognition.faces import FaceCatalog
    from tamis.recognition.worker import FaceDetectionWorker

    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=1)
    catalog = FaceCatalog()
    catalog.load(photos)

    called = []
    worker = FaceDetectionWorker(catalog, photos / "img000.jpg")
    worker.catalog = type("Spy", (), {"faces_for": lambda self, p: called.append(p) or []})()
    worker.cancel()
    worker.run()

    assert called == []


def test_cancelling_detection_leaves_no_worker_stranded_in_the_pending_list(main_window, tmp_path, qapp):
    # _pending_face_workers is what keeps each worker's signals object alive,
    # so an entry that never completes is both a leak and a lie about what is
    # still running. Dropping queued runnables from the pool would create
    # exactly that, which is why cancellation lets them run and exit instead.
    photos = tmp_path / "photos"
    photos.mkdir()
    _make_photos(photos, count=4)
    main_window.open_folder(photos)

    ctl = main_window.face_ctl
    ctl._active = True
    ctl.request_detection()
    ctl._warm_queue.extend(item.path for item in main_window.library.items)
    ctl._start_pending_detection()
    assert ctl._pending_face_workers

    ctl.cancel_detection_work()
    ctl.wait_for_detection_to_stop()
    deadline = time.time() + 20
    while ctl._pending_face_workers and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert ctl._pending_face_workers == []
