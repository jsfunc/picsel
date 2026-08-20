"""Owns the current photo's face-detection results and everything that
mutates them: the person gallery, the per-folder face cache, labeling,
Manage People, Forget All Faces, and keeping FacePanel/ImageViewer's face-box
overlay in sync.

Deliberately knows nothing about editing: the one place the two features
interact (Crop mode and Edit Faces mode can't both be active; overwriting a
photo invalidates its cached face data) is resolved by MainWindow, which
owns both controllers and is the natural place for cross-feature glue to
live.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from PySide6.QtCore import QRect, Qt, QThreadPool, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox

from tamis.recognition import FaceCatalog, PersonGallery
from tamis.recognition.worker import FaceDetectionWorker, SaveWorker
from tamis.views.face_panel import THUMBNAIL_SIZE, FaceEntry
from tamis.views.manage_people_dialog import ManagePeopleDialog

IMAGE_LOAD_PRIORITY = 10  # matches main_window.IMAGE_LOAD_PRIORITY -- see there for why

# Priorities within _detection_pool. The pool runs one job at a time, so these
# strictly order its queue: the photo being looked at always jumps ahead of
# any speculative warming still waiting.
_CURRENT_PHOTO_PRIORITY = 10
_WARM_PRIORITY = 0

# How long navigation has to settle before detection is requested for the
# photo landed on. Arrowing through photos faster than this enqueues no
# foreground work at all -- the same debounce shape as _face_filter_timer
# below and EditController._adjustment_timer.
_DETECTION_DEBOUNCE_MS = 200

# Upper bound on speculative warming jobs kept queued. Detection is a few
# hundred ms each, so an unbounded queue would let a long browse commit the
# GPU to minutes of work for photos the user has already left behind. The
# most recently visited are kept (a deque drops from the opposite end), since
# those are the ones nearest to where the user actually is.
_MAX_WARM_QUEUE = 32


class FaceRecognitionController:
    def __init__(self, parent_widget, library, viewer, face_panel, thread_pool) -> None:
        self.parent_widget = parent_widget  # for QMessageBox parenting
        self.library = library
        self.viewer = viewer
        self.face_panel = face_panel
        self.thread_pool = thread_pool
        self._active = False  # whether the Face Recognition tab is the one currently showing

        self.face_catalog = FaceCatalog()
        self.person_gallery = PersonGallery()
        if self.person_gallery.load_error:
            # Surfaced now, before anything can call save() and overwrite
            # the unreadable file with an empty gallery -- see
            # PersonGallery.load_error's docstring.
            QMessageBox.warning(self.parent_widget, "Face Gallery", self.person_gallery.load_error)

        # A separate, single-threaded pool (not the shared self.thread_pool
        # used for cancellable decode/detection work): person_gallery.save()
        # / face_catalog.save() rewrite their *entire* file on every call,
        # which can take hundreds of ms once a folder/gallery has any real
        # size to it -- doing that on the UI thread made every single face
        # confirmation feel laggy. maxThreadCount=1 keeps writes to the same
        # file strictly ordered (no risk of an older snapshot's write
        # finishing after a newer one's and clobbering it); waitForDone() on
        # this pool (not .clear() -- a queued save must never be dropped) is
        # how MainWindow.closeEvent guarantees the last write actually lands
        # before the app exits.
        self._save_thread_pool = QThreadPool()
        self._save_thread_pool.setMaxThreadCount(1)
        self._pending_save_workers: list[SaveWorker] = []

        # Detection runs on its own single-threaded pool, not the shared one.
        # Two reasons. It is single-threaded anyway (a shared nn.Module's
        # forward pass is serialized by the model lock in detector.py), so
        # extra threads buy nothing. And on the shared pool it actively
        # starved the rest of the app: browsing past 24 photos queued 24
        # workers, which occupied all 16 shared threads while blocked on that
        # lock, so decode work for the photo actually on screen had no thread
        # to run on -- measured at 152ms -> 2103ms for the displayed image.
        # Thread-pool priority could not help, since it orders the queue but
        # cannot preempt a running task.
        self._detection_pool = QThreadPool()
        self._detection_pool.setMaxThreadCount(1)
        self._pending_face_workers: list[FaceDetectionWorker] = []
        # Photos navigated past whose faces aren't cached yet, queued behind
        # the current photo to warm the cache. Bounded; see _MAX_WARM_QUEUE.
        self._warm_queue: deque[Path] = deque(maxlen=_MAX_WARM_QUEUE)
        self._queued_paths: set[Path] = set()  # in-flight/queued, to avoid detecting a photo twice
        self._last_requested_path: Path | None = None
        self._detection_timer = QTimer(parent_widget)
        self._detection_timer.setSingleShot(True)
        self._detection_timer.setInterval(_DETECTION_DEBOUNCE_MS)
        self._detection_timer.timeout.connect(self._start_pending_detection)
        # The records for whichever photo is currently displayed, so the
        # threshold slider can re-filter/redraw instantly without ever
        # calling back into FaceCatalog (which would re-run detection
        # synchronously on the UI thread if the cache weren't warm yet).
        self._current_face_path: Path | None = None
        self._current_face_records: list = []
        self._current_visible_face_records: list = []  # index-addressable, matches viewer's box order
        self._current_qimage: QImage | None = None  # for cropping face thumbnails
        self._current_qimage_path: Path | None = None
        # Coalesces bursts of threshold-slider ticks (one per pixel of drag)
        # into a bounded redraw rate -- _update_face_display() rebuilds
        # every visible face row and its full ranked person dropdown, which
        # otherwise reran on every single tick.
        self._face_filter_timer = QTimer(parent_widget)
        self._face_filter_timer.setSingleShot(True)
        self._face_filter_timer.setInterval(30)
        self._face_filter_timer.timeout.connect(self._update_face_display)

        face_panel.threshold_changed.connect(self._on_face_filter_changed)
        # edit_mode_toggled is deliberately NOT connected here: MainWindow
        # connects it to a small wrapper that also exits crop mode first
        # (the two modes can't both be active, since both interpret mouse
        # drags on the shared ImageViewer), then calls set_face_edit_mode
        # below itself.
        face_panel.name_confirmed.connect(self._on_face_name_confirmed)
        face_panel.remove_requested.connect(self._on_face_remove_requested)
        # manage_people_requested is deliberately NOT connected here: a
        # merge/forget can remove a Person that a SearchPanel scan (a
        # separate feature this controller knows nothing about) currently
        # holds a reference to, so MainWindow connects it to a wrapper that
        # calls show_manage_people_dialog() below and then cancels any
        # in-flight search targeting a removed person.
        face_panel.forget_all_requested.connect(self.forget_all)
        viewer.face_box_added.connect(self._on_face_box_added)
        viewer.face_box_dismiss_requested.connect(self._on_face_remove_requested)

    # -- Folder/persistence lifecycle, called by MainWindow -----------------

    def save_before_switching_folder(self) -> None:
        if self.face_catalog.folder is not None:
            self.save_face_catalog()

    def load_folder(self, folder: Path) -> None:
        # Anything still queued belongs to the folder being left. Its results
        # would be discarded anyway (faces_for's generation check refuses to
        # cache them into the new folder), so running them would just occupy
        # the GPU while the new folder's photos wait.
        self.cancel_detection_work()
        self.face_catalog.load(folder)
        if self.face_catalog.load_error:
            QMessageBox.warning(self.parent_widget, "Face Data", self.face_catalog.load_error)
            return  # nothing was loaded, so there is nothing to reconcile
        if self.person_gallery.load_error:
            # The gallery is empty only because its file couldn't be read, not
            # because there is genuinely nobody in it. Reconciling against it
            # would read every label in this folder as naming a person who no
            # longer exists, clear them all, and -- since a change triggers a
            # save -- write that over the folder's face data permanently. Same
            # reasoning as every other load_error guard in the app: never let
            # a failed read become a destructive write.
            return
        # Apply any merges/forgets done while this folder was closed. Without
        # this they only ever reached the folder that was open at the time,
        # leaving records here naming a person the gallery no longer has.
        if self.face_catalog.reconcile_people(
            self.person_gallery.merged_ids, {person.id for person in self.person_gallery.people}
        ):
            self.save_face_catalog()

    def save_face_catalog(self) -> None:
        # Snapshotting happens synchronously (prepare_save(), cheap -- a
        # simple list comprehension); only the slow JSON-serialize+write
        # step is deferred to _save_thread_pool. Safe even right before a
        # folder switch: FaceCatalog.load() *reassigns* self._records rather
        # than mutating it, so this snapshot is already immutable local data
        # by the time that happens.
        prepared = self.face_catalog.prepare_save()
        if prepared is None:
            return
        path, data = prepared
        worker = SaveWorker(self.face_catalog.write_payload, path, data)
        # Keep a Python reference until it finishes -- same reasoning as
        # ThumbnailList._pending_workers: with no reference held here, the
        # local `worker` var goes out of scope the instant this method
        # returns, letting `worker.signals` get garbage-collected before its
        # queued cross-thread `finished` emit is actually delivered.
        self._pending_save_workers.append(worker)
        worker.signals.finished.connect(
            lambda error, w=worker: self._on_save_finished(w, "Could not save face data", error)
        )
        self._save_thread_pool.start(worker)

    def save_person_gallery(self) -> None:
        path, data = self.person_gallery.prepare_save()
        worker = SaveWorker(self.person_gallery.write_payload, path, data)
        self._pending_save_workers.append(worker)
        worker.signals.finished.connect(
            lambda error, w=worker: self._on_save_finished(w, "Could not save the people gallery", error)
        )
        self._save_thread_pool.start(worker)

    def _on_save_finished(self, worker: SaveWorker, failure_context: str, error: str) -> None:
        if worker in self._pending_save_workers:
            self._pending_save_workers.remove(worker)
        if error:
            QMessageBox.warning(self.parent_widget, "Save Failed", f"{failure_context}:\n{error}")

    def wait_for_pending_saves(self) -> None:
        """Block until every queued/in-flight face-catalog or person-gallery
        write has actually completed -- called by MainWindow.closeEvent right
        before the app exits, so a save() triggered moments before closing
        (e.g. the last face confirmed) can't get silently dropped."""
        self._save_thread_pool.waitForDone()

    # -- Tab/navigation lifecycle, called by MainWindow ----------------------

    def on_tab_activated(self) -> None:
        self._active = True
        self.request_detection()

    def on_tab_deactivated(self) -> None:
        self._active = False
        self.exit_face_edit_mode()

    def exit_face_edit_mode(self) -> None:
        self.viewer.set_face_edit_mode(False)
        self.face_panel.set_edit_mode_active(False)

    def on_new_photo_loaded(self, qimage: QImage, path: Path) -> None:
        """Called once a newly-navigated-to photo's full-resolution image has
        finished loading."""
        # Symmetric with EditController's crop-mode reset: a face box drawn
        # against the previous photo shouldn't silently carry over and get
        # attached to whichever photo happens to load next.
        self.exit_face_edit_mode()
        self._current_qimage = qimage
        self._current_qimage_path = path
        # Face detection runs concurrently and can finish before this
        # image-load worker does; if it already has, set_image() just wiped
        # its overlay, so redraw it now against this (correct, just-arrived)
        # image instead of leaving no boxes shown at all.
        if self._current_face_path == path:
            self._update_face_display()

    def invalidate(self, path: Path) -> None:
        self.face_catalog.invalidate(path)

    def invalidate_and_maybe_redetect(self, path: Path) -> None:
        """Call after overwriting `path` on disk -- the cached boxes/
        embeddings were computed against the pre-edit pixel geometry and are
        now wrong."""
        self.face_catalog.invalidate(path)
        if self._current_face_path == path:
            self.request_detection()

    def invalidate_moved(self, item, original_path: Path) -> None:
        """Call for each item Apply Culling attempted to move, *before*
        saving the face catalog -- drops cached face data for anything that
        actually left this folder, so the save doesn't write a now-orphaned
        entry into this folder's .tamis_faces.json (ready to misattribute
        results if a future photo reuses the same filename here)."""
        if item.path != original_path:
            self.face_catalog.invalidate(original_path)

    # -- Detection ------------------------------------------------------

    def request_detection(self) -> None:
        """Kick off (or re-display cached) face detection for the current photo.

        No-op if the Face Recognition tab isn't the active one -- browsing
        without it selected shouldn't pay detection's per-photo cost (a few
        hundred ms on this machine's GPU, more on CPU) for a feature the user
        isn't looking at.
        """
        if not self._active:
            return
        item = self.library.current_item
        # Cleared immediately, not after the debounce: whatever is on screen
        # belongs to the photo being navigated away from, and must not appear
        # to describe the new one even briefly.
        self._current_face_path = None
        self._current_face_records = []
        self._current_visible_face_records = []
        self.viewer.set_face_boxes([])
        self.face_panel.set_faces([])
        if item is None:
            self._detection_timer.stop()
            self.face_panel.set_status("No photo open.")
            return

        # The photo just left, if it never got detected, becomes a candidate
        # for speculative warming rather than being abandoned -- its result is
        # cached per-photo, so the work stays useful if the user comes back.
        if (
            self._last_requested_path is not None
            and self._last_requested_path != item.path
            and self._last_requested_path not in self._queued_paths
            and not self.face_catalog.is_cached(self._last_requested_path)
        ):
            self._warm_queue.append(self._last_requested_path)
        self._last_requested_path = item.path

        self.face_panel.set_status(f"Detecting faces in {item.name}...")
        # Debounced: arrowing through photos faster than this enqueues no
        # foreground detection for the ones passed through, which is what
        # kept the photo finally landed on waiting behind every photo skimmed
        # to reach it (measured at 3.3s for the 12th of 12).
        self._detection_timer.start()

    def _start_pending_detection(self) -> None:
        """Fire the debounced request: the current photo first, then whatever
        speculative warming has accumulated behind it."""
        item = self.library.current_item
        if item is None or not self._active:
            return
        self._start_detection(item.path, _CURRENT_PHOTO_PRIORITY)
        while self._warm_queue:
            self._start_detection(self._warm_queue.popleft(), _WARM_PRIORITY)

    def _start_detection(self, path: Path, priority: int) -> None:
        if path in self._queued_paths:
            return  # already queued or running; detecting it twice caches the same result twice
        self._queued_paths.add(path)
        worker = FaceDetectionWorker(self.face_catalog, path)
        # Kept alive until it finishes, same reasoning as MainWindow's other
        # pending-worker lists: otherwise its signals QObject could be
        # garbage-collected mid-run.
        self._pending_face_workers.append(worker)
        worker.signals.finished.connect(
            lambda p, records, error, w=worker: self._on_faces_detected(p, records, error, w)
        )
        self._detection_pool.start(worker, priority)

    def cancel_detection_work(self) -> None:
        """Abandon everything queued or running, without waiting for it.

        Queued workers exit at the top of run(); one already past that point
        finishes, which is bounded by a single photo's detection. Used on a
        folder switch (its results would be discarded anyway, and worse, could
        be cached against the wrong folder) and on quit, so a browse that left
        a warming queue behind can't hold the app open.

        Deliberately does *not* call `_detection_pool.clear()`. Dropping a
        queued runnable means it never runs, so it never emits `finished` and
        never gets removed from `_pending_face_workers` -- it would sit there
        for the life of the process, and it is exactly the reference that
        keeps its signals object alive. Letting every worker run and exit
        immediately costs microseconds each and keeps that bookkeeping honest.
        """
        for worker in self._pending_face_workers:
            worker.cancel()
        self._warm_queue.clear()
        self._queued_paths.clear()
        self._last_requested_path = None
        self._detection_timer.stop()

    def wait_for_detection_to_stop(self) -> None:
        self._detection_pool.waitForDone()

    def _on_faces_detected(self, path: Path, records: list, error: str, worker) -> None:
        if worker in self._pending_face_workers:
            self._pending_face_workers.remove(worker)
        self._queued_paths.discard(path)
        if worker.cancelled:
            return  # abandoned by a folder switch or a quit; `records` is empty
        item = self.library.current_item
        if item is None or path != item.path:
            return  # user has navigated to a different photo since this was requested
        if error:
            self.face_panel.set_status(f"Face detection failed: {error}")
            return
        self._current_face_path = path
        self._current_face_records = records
        self._update_face_display()

    def _on_face_filter_changed(self, _value: float) -> None:
        # Pure re-filter over already-cached records -- never touches
        # FaceCatalog or re-runs the model, so it's safe to run on every
        # slider tick correctness-wise. Still debounced (see
        # _face_filter_timer) since the redraw itself -- rebuilding every
        # visible face row and its full ranked person dropdown -- isn't
        # free, and doing that on every pixel of drag is wasted work.
        self._face_filter_timer.start()

    # -- Manual add/remove ------------------------------------------------

    def set_face_edit_mode(self, enabled: bool) -> None:
        self.viewer.set_face_edit_mode(enabled)

    def _on_face_box_added(self, box: tuple[int, int, int, int]) -> None:
        item = self.library.current_item
        if item is None:
            return
        record = self.face_catalog.add_manual_face(item.path, box)
        if self._current_face_path == item.path:
            self._current_face_records.append(record)
        self.save_face_catalog()
        self._update_face_display()

    def _on_face_remove_requested(self, index: int) -> None:
        if index < 0 or index >= len(self._current_visible_face_records):
            return
        item = self.library.current_item
        if item is None:
            return
        record = self._current_visible_face_records[index]
        if record.is_manual:
            self.face_catalog.remove_manual_face(item.path, record)
            if record in self._current_face_records:
                self._current_face_records.remove(record)
        else:
            self.face_catalog.dismiss(record)
        self.save_face_catalog()
        self._update_face_display()

    # -- Labeling ---------------------------------------------------------

    def _on_face_name_confirmed(self, index: int, name: str) -> None:
        if index < 0 or index >= len(self._current_visible_face_records):
            return
        record = self._current_visible_face_records[index]
        current_person = self.person_gallery.find_by_id(record.person_id) if record.person_id else None
        if record.person_id is not None and current_person is None:
            # A label naming someone the gallery no longer has (forgotten, or
            # merged away before `merged_ids` recorded redirects). Clear it
            # explicitly instead of falling through with it still set: the
            # branch below only removes the old sample when `current_person`
            # is found, so leaving a stale id here is what used to let a
            # re-confirmation add a *second* copy of this face's sample while
            # the orphaned first copy stayed put.
            self.face_catalog.assign_person(record, None)
        current_name = current_person.name if current_person is not None else ""
        if name == current_name:
            return  # re-confirming an unchanged name -- don't add a duplicate embedding sample

        if current_person is not None:
            # Being relabeled to someone else, or unassigned: the sample
            # added under the old label no longer describes this face, so it
            # shouldn't keep sitting in that person's reference data.
            self.person_gallery.remove_embedding(current_person.id, record.embedding)

        if name:
            person = self.person_gallery.find_by_name(name) or self.person_gallery.add_person(name)
            self.person_gallery.add_embedding(person.id, record.embedding)
            self.face_catalog.assign_person(record, person.id)
        else:
            self.face_catalog.assign_person(record, None)

        self.save_person_gallery()
        self.save_face_catalog()
        self._update_face_display()

    def show_manage_people_dialog(self) -> set[str]:
        """Returns the ids of any people removed by a merge or forget, so a
        caller that needs to react to a Person disappearing (e.g. cancel a
        SearchPanel scan holding a reference to it) can do so."""
        dialog = ManagePeopleDialog(self.person_gallery, self.parent_widget)
        dialog.exec()
        # Only the currently-loaded folder's face records can be updated
        # here; a merged-away/forgotten person's label in a folder that isn't
        # open right now will just show as unconfirmed next time that folder
        # is opened (for a merge, its correct suggestion should resurface on
        # its own, since the same embeddings now live under the kept person).
        removed_ids = {removed_id for removed_id, _kept_id in dialog.merges} | set(dialog.forgotten_ids)
        if removed_ids:
            for removed_id, kept_id in dialog.merges:
                self.face_catalog.remap_person(removed_id, kept_id)
            for forgotten_id in dialog.forgotten_ids:
                self.face_catalog.forget_person(forgotten_id)
            self.save_face_catalog()
            self._update_face_display()
        return removed_ids

    def forget_all(self) -> None:
        if not self.person_gallery.people:
            return
        confirm = QMessageBox.question(
            self.parent_widget,
            "Forget All Faces",
            "Forget every named person and all their recognition data? This cannot be undone. "
            "Faces already labeled in photos will show as unconfirmed again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.person_gallery.people = []
        self.save_person_gallery()
        self.face_catalog.unassign_all_people()
        self.save_face_catalog()
        self._update_face_display()

    # -- Display ------------------------------------------------------

    def _update_face_display(self) -> None:
        item = self.library.current_item
        if item is None or self._current_face_path != item.path:
            return
        threshold = self.face_panel.threshold()
        visible = [
            record
            for record in self._current_face_records
            if not record.dismissed and (record.is_manual or record.confidence >= threshold)
        ]
        self._current_visible_face_records = visible
        self.viewer.set_face_boxes([record.box for record in visible])
        self.face_panel.set_faces([self._build_face_entry(record) for record in visible])
        self.face_panel.set_status(
            f"{item.name}: {len(visible)} face(s) shown ({len(self._current_face_records)} candidate(s) total)"
        )

    def _build_face_entry(self, record) -> FaceEntry:
        # The single best guess, from the k-nearest-samples neighborhood
        # (more robust than one lucky/unlucky sample) -- always shown (no
        # threshold gate), its similarity conveyed to the user via color.
        best_guess = self.person_gallery.identify(record.embedding)
        top_suggestion = best_guess[0][0].name if best_guess else ""
        top_suggestion_similarity = best_guess[0][1] if best_guess else 0.0

        # Every known person, not just the best-guess neighborhood: lets the
        # user manually pick someone the k-nearest cutoff excluded.
        dropdown_entries = [
            (person.name, similarity) for person, similarity in self.person_gallery.rank_all(record.embedding)
        ]

        confirmed_name = ""
        if record.person_id is not None:
            person = self.person_gallery.find_by_id(record.person_id)
            confirmed_name = person.name if person is not None else ""
            if confirmed_name and confirmed_name not in (name for name, _similarity in dropdown_entries):
                dropdown_entries.insert(0, (confirmed_name, 1.0))

        return FaceEntry(
            thumbnail=self._crop_face_thumbnail(record.box),
            dropdown_entries=dropdown_entries,
            top_suggestion=top_suggestion,
            top_suggestion_similarity=top_suggestion_similarity,
            confirmed_name=confirmed_name,
            is_manual=record.is_manual,
        )

    def _crop_face_thumbnail(self, box: tuple[int, int, int, int]) -> QPixmap:
        current_item = self.library.current_item
        if self._current_qimage is None or current_item is None or self._current_qimage_path != current_item.path:
            return QPixmap()  # image for this photo hasn't finished loading yet
        left, top, right, bottom = box
        left = max(0, left)
        top = max(0, top)
        right = min(self._current_qimage.width(), right)
        bottom = min(self._current_qimage.height(), bottom)
        if right <= left or bottom <= top:
            return QPixmap()
        cropped = self._current_qimage.copy(QRect(left, top, right - left, bottom - top))
        return QPixmap.fromImage(cropped).scaled(
            THUMBNAIL_SIZE,
            THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
