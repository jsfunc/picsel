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

from pathlib import Path

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox

from picsel.recognition import FaceCatalog, PersonGallery
from picsel.recognition.worker import FaceDetectionWorker
from picsel.views.face_panel import THUMBNAIL_SIZE, FaceEntry
from picsel.views.manage_people_dialog import ManagePeopleDialog

IMAGE_LOAD_PRIORITY = 10  # matches main_window.IMAGE_LOAD_PRIORITY -- see there for why


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

        self._pending_face_workers: list[FaceDetectionWorker] = []
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
        self.face_catalog.load(folder)
        if self.face_catalog.load_error:
            QMessageBox.warning(self.parent_widget, "Face Data", self.face_catalog.load_error)

    def save_face_catalog(self) -> None:
        try:
            self.face_catalog.save()
        except OSError as exc:
            QMessageBox.warning(self.parent_widget, "Save Failed", f"Could not save face data:\n{exc}")

    def save_person_gallery(self) -> None:
        try:
            self.person_gallery.save()
        except OSError as exc:
            QMessageBox.warning(self.parent_widget, "Save Failed", f"Could not save the people gallery:\n{exc}")

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
        entry into this folder's .picsel_faces.json (ready to misattribute
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
        self._current_face_path = None
        self._current_face_records = []
        self._current_visible_face_records = []
        self.viewer.set_face_boxes([])
        self.face_panel.set_faces([])
        if item is None:
            self.face_panel.set_status("No photo open.")
            return

        self.face_panel.set_status(f"Detecting faces in {item.name}...")
        worker = FaceDetectionWorker(self.face_catalog, item.path)
        # Kept alive until it finishes, same reasoning as MainWindow's other
        # pending-worker lists: otherwise its signals QObject could be
        # garbage-collected mid-run.
        self._pending_face_workers.append(worker)
        worker.signals.finished.connect(
            lambda path, records, error, w=worker: self._on_faces_detected(path, records, error, w)
        )
        # Same elevated priority as the visible full-image load -- this is
        # just as much "what the user is looking at right now" as the image
        # itself, and without it, switching to the Face Recognition tab
        # right after opening a large never-before-processed folder could
        # queue behind thousands of pending thumbnail jobs before detection
        # for the current photo even starts.
        self.thread_pool.start(worker, IMAGE_LOAD_PRIORITY)

    def _on_faces_detected(self, path: Path, records: list, error: str, worker) -> None:
        if worker in self._pending_face_workers:
            self._pending_face_workers.remove(worker)
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
