"""Main application window: layout, menus, shortcuts, and controller logic."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QRect, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
)

from picsel import __version__
from picsel.editing import EditSession
from picsel.io_ops import (
    apply_culling,
    capture_time,
    find_sequence_groups,
    rename_by_creation_date,
    renumber_by_creation_time,
    rename_with_sequence,
)
from picsel.models import ImageLibrary, Status
from picsel.thumbnails import ImageLoadWorker, MetadataLoadWorker, pil_to_qimage
from picsel.views.dialogs import ApplyCullingDialog, RenameDialog, RenumberDialog
from picsel.views.edit_panel import EditPanel
from picsel.views.image_viewer import ImageViewer
from picsel.views.metadata_panel import MetadataPanel
from picsel.views.thumbnail_list import ThumbnailList

# Face recognition depends on torch/facenet-pytorch, an optional heavy extra
# (see requirements-recognition.txt) not installed by the base ./install.sh --
# the whole feature degrades to "not available" rather than breaking the app
# for anyone who hasn't opted into it.
try:
    from picsel.recognition import FaceCatalog, PersonGallery
    from picsel.recognition.worker import FaceDetectionWorker
    from picsel.views.face_panel import THUMBNAIL_SIZE, FaceEntry, FacePanel
    from picsel.views.manage_people_dialog import ManagePeopleDialog
    from picsel.views.search_panel import SearchPanel

    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False

IMAGE_LOAD_PRIORITY = 10  # above the default (0) used by thumbnail workers

SHORTCUTS_TEXT = """\
Navigation
  Right / D       Next image
  Left  / A       Previous image

Culling
  S / Up          Mark Selected
  X / Down        Mark Rejected
  U               Unmark
  1-5             Set star rating
  0               Clear rating

Editing
  E               Show Edit Image panel
  M               Show Image Information panel
  R               Rotate clockwise
  Shift+R         Rotate counter-clockwise
  H               Flip horizontal
  V               Flip vertical
  Ctrl+Z          Undo edit
  Ctrl+Shift+Z    Redo edit
  Ctrl+S          Save edited copy

Library
  Ctrl+Shift+A    Apply Culling (move/copy to folders)
  N               Rename with name + sequence number
  Ctrl+Shift+N    Renumber a sequence by creation time
  Ctrl+Shift+D    Rename all by creation date (pYYYYmmdd_hhmmss.ext)
"""

if RECOGNITION_AVAILABLE:
    SHORTCUTS_TEXT = SHORTCUTS_TEXT.replace(
        "  M               Show Image Information panel\n",
        "  M               Show Image Information panel\n  F               Show Face Recognition panel\n",
    )


def _bundled_resource_path(relative: str) -> Path:
    """Resolve a bundled data file's path, whether running from source or as
    a frozen PyInstaller executable -- picsel.spec's one-file build extracts
    its `datas` into a temp dir at runtime, referenced by `sys._MEIPASS`,
    rather than leaving them alongside the source tree.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parent.parent  # repo root
    return base / relative


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"picSel {__version__}")
        self.resize(1280, 860)

        self.library = ImageLibrary()
        self.edit_session: EditSession | None = None
        self._thread_pool = QThreadPool.globalInstance()
        self._sort_mode = "name"  # persists across folder switches within the session

        self._image_load_generation = 0
        self._pending_image_workers: list[ImageLoadWorker] = []

        self._metadata_load_generation = 0
        self._pending_metadata_workers: list[MetadataLoadWorker] = []

        if RECOGNITION_AVAILABLE:
            self.face_catalog = FaceCatalog()
            self.person_gallery = PersonGallery()
            if self.person_gallery.load_error:
                # Surfaced now, before anything can call save() and overwrite
                # the unreadable file with an empty gallery -- see
                # PersonGallery.load_error's docstring.
                QMessageBox.warning(self, "Face Gallery", self.person_gallery.load_error)
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
            # Coalesces bursts of threshold-slider ticks (one per pixel of
            # drag) into a bounded redraw rate, same reasoning as
            # _adjustment_timer below -- _update_face_display() rebuilds
            # every visible face row and its full ranked person dropdown,
            # which otherwise reran on every single tick.
            self._face_filter_timer = QTimer(self)
            self._face_filter_timer.setSingleShot(True)
            self._face_filter_timer.setInterval(30)
            self._face_filter_timer.timeout.connect(self._update_face_display)

        self._pending_adjustments: tuple[float, float, float] | None = None
        self._adjustment_timer = QTimer(self)
        self._adjustment_timer.setSingleShot(True)
        self._adjustment_timer.setInterval(30)
        self._adjustment_timer.timeout.connect(self._apply_pending_adjustments)

        self.viewer = ImageViewer()
        self.thumbnail_list = ThumbnailList()
        self.thumbnail_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.edit_panel = EditPanel()
        self.metadata_panel = MetadataPanel()

        # One panel visible at a time, selected by tab, rather than several
        # independently-toggleable docks -- Image Information is the default.
        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.metadata_panel, "Image Information")
        self.side_tabs.addTab(self.edit_panel, "Edit Image")

        # Metadata/edit/faces sit beside the image only (not the full window
        # height like a dock would), so the thumbnail strip below spans the
        # full width.
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.viewer)
        top_splitter.addWidget(self.side_tabs)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 0)
        top_splitter.setSizes([900, 450])

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_splitter)
        splitter.addWidget(self.thumbnail_list)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 150])
        self.setCentralWidget(splitter)

        self.side_tabs.currentChanged.connect(self._on_side_tab_changed)

        if RECOGNITION_AVAILABLE:
            self.face_panel = FacePanel()
            self.side_tabs.addTab(self.face_panel, "Face Recognition")
            self.face_panel.threshold_changed.connect(self._on_face_filter_changed)
            self.face_panel.edit_mode_toggled.connect(self._on_face_edit_mode_toggled)
            self.face_panel.name_confirmed.connect(self._on_face_name_confirmed)
            self.face_panel.remove_requested.connect(self._on_face_remove_requested)
            self.face_panel.manage_people_requested.connect(self._show_manage_people_dialog)
            self.face_panel.forget_all_requested.connect(self._on_forget_all_faces)
            self.viewer.face_box_added.connect(self._on_face_box_added)
            self.viewer.face_box_dismiss_requested.connect(self._on_face_remove_requested)

            self.search_panel = SearchPanel(
                self.library, self.face_catalog, self.person_gallery, self._thread_pool, self.face_panel.threshold
            )
            self.side_tabs.addTab(self.search_panel, "Search by Name")
            self.search_panel.photo_chosen.connect(self._on_search_photo_chosen)

        self._build_menu()
        self._build_shortcuts()

        self.thumbnail_list.currentRowChanged.connect(self._on_thumbnail_selected)
        self.viewer.crop_selected.connect(self._on_crop_selected)
        self._connect_edit_panel()

        self._update_status_bar()

    # -- Menu / shortcuts --------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open Folder...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._choose_folder)
        file_menu.addAction(open_action)

        apply_action = QAction("Apply Culling...", self)
        apply_action.setShortcut("Ctrl+Shift+A")
        apply_action.triggered.connect(self._apply_culling)
        file_menu.addAction(apply_action)

        rename_action = QAction("Rename...", self)
        rename_action.setShortcut("N")
        rename_action.triggered.connect(self._rename_current)
        file_menu.addAction(rename_action)

        renumber_action = QAction("Renumber by Creation Time...", self)
        renumber_action.setShortcut("Ctrl+Shift+N")
        renumber_action.triggered.connect(self._renumber_by_creation_time)
        file_menu.addAction(renumber_action)

        rename_by_date_action = QAction("Rename All by Creation Date...", self)
        rename_by_date_action.setShortcut("Ctrl+Shift+D")
        rename_by_date_action.triggered.connect(self._rename_by_creation_date)
        file_menu.addAction(rename_by_date_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        sort_group = QActionGroup(self)
        sort_group.setExclusive(True)

        self.sort_by_name_action = QAction("Sort by Name", self)
        self.sort_by_name_action.setCheckable(True)
        self.sort_by_name_action.setChecked(True)
        self.sort_by_name_action.triggered.connect(lambda: self._set_sort_mode("name"))
        sort_group.addAction(self.sort_by_name_action)
        view_menu.addAction(self.sort_by_name_action)

        self.sort_by_date_action = QAction("Sort by Date Taken", self)
        self.sort_by_date_action.setCheckable(True)
        self.sort_by_date_action.triggered.connect(lambda: self._set_sort_mode("date"))
        sort_group.addAction(self.sort_by_date_action)
        view_menu.addAction(self.sort_by_date_action)

        self.sort_by_stars_action = QAction("Sort by Star Rating", self)
        self.sort_by_stars_action.setCheckable(True)
        self.sort_by_stars_action.triggered.connect(lambda: self._set_sort_mode("stars"))
        sort_group.addAction(self.sort_by_stars_action)
        view_menu.addAction(self.sort_by_stars_action)

        help_menu = self.menuBar().addMenu("&Help")
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

        if RECOGNITION_AVAILABLE:
            face_docs_action = QAction("Face Recognition Docs", self)
            face_docs_action.triggered.connect(self._open_face_recognition_docs)
            help_menu.addAction(face_docs_action)

        about_action = QAction("About picSel", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_shortcuts(self) -> None:
        def add(sequence: str, handler) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(handler)

        add("Right", self._go_next)
        add("D", self._go_next)
        add("Left", self._go_prev)
        add("A", self._go_prev)

        add("S", lambda: self._set_status(Status.SELECTED))
        add("Up", lambda: self._set_status(Status.SELECTED))
        add("X", lambda: self._set_status(Status.REJECTED))
        add("Down", lambda: self._set_status(Status.REJECTED))
        add("U", lambda: self._set_status(Status.UNRATED))

        for rating in range(1, 6):
            add(str(rating), lambda r=rating: self._set_rating(r))
        add("0", lambda: self._set_rating(0))

        add("E", lambda: self.side_tabs.setCurrentWidget(self.edit_panel))
        add("M", lambda: self.side_tabs.setCurrentWidget(self.metadata_panel))
        if RECOGNITION_AVAILABLE:
            add("F", lambda: self.side_tabs.setCurrentWidget(self.face_panel))
        add("R", self.edit_panel.rotate_cw.emit)
        add("Shift+R", self.edit_panel.rotate_ccw.emit)
        add("H", self.edit_panel.flip_horizontal.emit)
        add("V", self.edit_panel.flip_vertical.emit)
        add("Ctrl+Z", self.edit_panel.undo_requested.emit)
        add("Ctrl+Shift+Z", self.edit_panel.redo_requested.emit)
        add("Ctrl+S", self.edit_panel.save_copy_requested.emit)

    def _show_shortcuts(self) -> None:
        QMessageBox.information(self, "Keyboard Shortcuts", SHORTCUTS_TEXT)

    def _open_face_recognition_docs(self) -> None:
        path = _bundled_resource_path("docs/face_recognition.html")
        if not path.exists():
            QMessageBox.warning(self, "Face Recognition Docs", f"Documentation file not found:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _show_about(self) -> None:
        recognition_line = (
            "Face recognition: enabled" if RECOGNITION_AVAILABLE else "Face recognition: not installed"
        )
        QMessageBox.about(
            self,
            "About picSel",
            f"<b>picSel</b> {__version__}<br><br>"
            "A small desktop app for culling and lightly editing a folder of photos.<br><br>"
            f"{recognition_line}<br><br>"
            'GPLv3 — <a href="https://github.com/jsfunc/picsel">github.com/jsfunc/picsel</a>',
        )

    # -- Folder / library ----------------------------------------------------

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self.open_folder(Path(folder))

    def open_folder(self, folder: Path) -> None:
        if not self._can_navigate_away():
            return
        if self.library.folder is not None:
            self._save_library_state()
        if RECOGNITION_AVAILABLE and self.face_catalog.folder is not None:
            self._save_face_catalog()

        self.edit_session = None
        try:
            self.library.load(folder)
        except OSError as exc:
            QMessageBox.critical(self, "Open Folder Failed", f"Could not read {folder}:\n{exc}")
            return
        if self.library.load_error:
            QMessageBox.warning(self, "Ratings/Status", self.library.load_error)
        if RECOGNITION_AVAILABLE:
            self.face_catalog.load(folder)
            if self.face_catalog.load_error:
                QMessageBox.warning(self, "Face Data", self.face_catalog.load_error)

        if self._sort_mode != "name" and self.library.items:
            self.library.sort_items(key=self._sort_key(self._sort_mode))

        self.thumbnail_list.set_items(self.library.items)
        self.setWindowTitle(f"picSel {__version__} — {folder}")

        if self.library.items:
            self.library.current_index = 0
            self._show_current()
        else:
            self.viewer.set_image(QImage())
            self.metadata_panel.set_image(None)
            self.statusBar().showMessage(f"No supported images found in {folder}")

    # -- Persistence helpers ---------------------------------------------
    # Every actual file operation elsewhere in this class (rename, save-as,
    # culling) is wrapped in try/except OSError with a message shown to the
    # user; these three wrap the equivalent for the app's own state files
    # (ratings/status, per-folder face cache, person gallery) so a folder on
    # removable/network media going unwritable mid-session reports a clear
    # error instead of raising out of whatever keyboard shortcut or signal
    # handler happened to trigger the save.

    def _save_library_state(self) -> None:
        try:
            self.library.save_state()
        except OSError as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not save photo ratings/status:\n{exc}")

    def _save_face_catalog(self) -> None:
        try:
            self.face_catalog.save()
        except OSError as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not save face data:\n{exc}")

    def _save_person_gallery(self) -> None:
        try:
            self.person_gallery.save()
        except OSError as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not save the people gallery:\n{exc}")

    # -- Navigation ----------------------------------------------------

    def _can_navigate_away(self) -> bool:
        if self.edit_session is not None and self.edit_session.has_edits():
            reply = QMessageBox.question(
                self,
                "Discard edits?",
                f"Discard unsaved edits to {self.edit_session.source_path.name}?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Discard:
                return False
            self.edit_session = None
        return True

    def _go_next(self) -> None:
        if not self._can_navigate_away():
            return
        self.library.next()
        self._show_current()

    def _go_prev(self) -> None:
        if not self._can_navigate_away():
            return
        self.library.prev()
        self._show_current()

    def _on_thumbnail_selected(self, index: int) -> None:
        if index < 0 or index == self.library.current_index:
            return
        if not self._can_navigate_away():
            self.thumbnail_list.select_index(self.library.current_index)
            return
        self.library.current_index = index
        self._show_current()

    def _show_current(self) -> None:
        item = self.library.current_item
        if item is None:
            self.viewer.set_image(QImage())
            self.metadata_panel.set_image(None)
            self._update_status_bar()
            return

        # Metadata reflects the on-disk file regardless of any pending unsaved
        # edits, and doesn't need to wait on the (async) full-resolution decode
        # -- but it's read asynchronously too now, same reasoning as the image
        # itself: extract_metadata() does file I/O + IFD parsing per call, and
        # running that on the UI thread on every single navigation could
        # visibly stutter rapid next/prev browsing.
        self._load_metadata_async(item.path)

        if self.edit_session is not None and self.edit_session.source_path == item.path:
            self._refresh_preview()
        else:
            self.edit_session = None
            self._load_image_async(item.path)

        if RECOGNITION_AVAILABLE:
            self._request_face_detection()

        self.thumbnail_list.select_index(self.library.current_index)
        self._update_status_bar()

    def _load_image_async(self, path: Path) -> None:
        self._image_load_generation += 1
        generation = self._image_load_generation
        worker = ImageLoadWorker(path)
        # Keep a reference until the worker actually finishes: it runs on a
        # background thread, and if this were the only Python reference to it,
        # a later navigation could let it (and its signals object) get garbage
        # collected mid-run, crashing the worker thread on emit().
        self._pending_image_workers.append(worker)
        worker.signals.finished.connect(
            lambda p, image, error, gen=generation, w=worker: self._on_image_loaded(gen, p, image, error, w)
        )
        # Higher priority than thumbnail workers (default 0, shared QThreadPool),
        # so the visible image doesn't wait behind a large folder's thumbnail queue.
        self._thread_pool.start(worker, IMAGE_LOAD_PRIORITY)

    def _on_image_loaded(
        self, generation: int, path: Path, qimage: QImage, error: str, worker: ImageLoadWorker
    ) -> None:
        if worker in self._pending_image_workers:
            self._pending_image_workers.remove(worker)
        if generation != self._image_load_generation:
            return
        if error:
            self.statusBar().showMessage(f"Failed to load {path.name}: {error}")
        self.viewer.set_image(qimage)  # also clears any face-box overlay from the previous photo
        self.edit_panel.set_image_size(qimage.width(), qimage.height())
        self.edit_panel.reset_adjustment_sliders()
        self.edit_panel.set_history_enabled(False, False)
        self.viewer.set_crop_mode(False)
        self.edit_panel.set_crop_mode_active(False)

        if RECOGNITION_AVAILABLE:
            # Symmetric with crop mode above: a face box drawn against the
            # previous photo shouldn't silently carry over and get attached
            # to whichever photo happens to load next.
            self.viewer.set_face_edit_mode(False)
            self.face_panel.set_edit_mode_active(False)
            self._current_qimage = qimage
            self._current_qimage_path = path
            # Face detection runs concurrently and can finish before this
            # image-load worker does; if it already has, set_image() above
            # just wiped its overlay, so redraw it now against this (correct,
            # just-arrived) image instead of leaving no boxes shown at all.
            if self._current_face_path == path:
                self._update_face_display()

    def _load_metadata_async(self, path: Path) -> None:
        self._metadata_load_generation += 1
        generation = self._metadata_load_generation
        worker = MetadataLoadWorker(path)
        # Kept alive until it finishes, same reasoning as _pending_image_workers.
        self._pending_metadata_workers.append(worker)
        worker.signals.finished.connect(
            lambda p, sections, error, gen=generation, w=worker: self._on_metadata_loaded(gen, p, sections, error, w)
        )
        self._thread_pool.start(worker)

    def _on_metadata_loaded(
        self, generation: int, path: Path, sections: list, error: str, worker: MetadataLoadWorker
    ) -> None:
        if worker in self._pending_metadata_workers:
            self._pending_metadata_workers.remove(worker)
        if generation != self._metadata_load_generation:
            return  # user has navigated to a different photo since this was requested
        if error:
            self.statusBar().showMessage(f"Failed to load metadata for {path.name}: {error}")
        self.metadata_panel.set_sections(sections)

    def _update_status_bar(self) -> None:
        item = self.library.current_item
        if item is None:
            self.statusBar().showMessage("Open a folder to get started (File > Open Folder)")
            return
        counts = self.library.counts()
        position = f"{self.library.current_index + 1}/{len(self.library.items)}"
        rating = "*" * item.rating if item.rating else "-"
        message = (
            f"{item.name}  |  {position}  |  Status: {item.status.value}  |  Rating: {rating}"
            f"  |  Selected: {counts['selected']}  Rejected: {counts['rejected']}  Unrated: {counts['unrated']}"
        )
        self.statusBar().showMessage(message)

    # -- Sorting ----------------------------------------------------

    def _sort_key(self, mode: str):
        if mode == "date":
            return lambda item: capture_time(item.path)
        if mode == "stars":
            # Highest rating first; break ties by capture time (earliest first).
            return lambda item: (-item.rating, capture_time(item.path))
        return lambda item: item.path

    def _set_sort_mode(self, mode: str) -> None:
        if mode == self._sort_mode:
            return
        self._sort_mode = mode
        # Keep the View menu in sync even when called programmatically (e.g.
        # reapplying the preference in open_folder): a menu click checks the
        # action before triggered() fires, but a direct call here never
        # touches the action's checked state unless we set it ourselves.
        sort_actions = {
            "name": self.sort_by_name_action,
            "date": self.sort_by_date_action,
            "stars": self.sort_by_stars_action,
        }
        sort_actions[mode].setChecked(True)
        if self.library.items:
            self.library.sort_items(key=self._sort_key(mode))
            self.thumbnail_list.set_items(self.library.items)
            self.thumbnail_list.select_index(self.library.current_index)
            self._update_status_bar()

    # -- Marking ----------------------------------------------------

    def _set_status(self, status: Status) -> None:
        if self.library.current_item is None:
            return
        self.library.set_status(self.library.current_index, status)
        self.thumbnail_list.refresh_badges()
        self._save_library_state()
        self._update_status_bar()

    def _set_rating(self, rating: int) -> None:
        if self.library.current_item is None:
            return
        self.library.set_rating(self.library.current_index, rating)
        self.thumbnail_list.refresh_badges()
        self._save_library_state()
        self._update_status_bar()

    # -- Rename ----------------------------------------------------

    def _rename_current(self) -> None:
        item = self.library.current_item
        if item is None:
            return
        if not self._can_navigate_away():
            return

        dialog = RenameDialog(list(self.library.renamed_names.keys()), self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.chosen_name:
            return

        name = dialog.chosen_name
        number = self.library.register_name_use(name)
        try:
            new_path = rename_with_sequence(item, name, number)
        except OSError as exc:
            QMessageBox.critical(self, "Rename Failed", str(exc))
            return

        self.edit_session = None
        self.thumbnail_list.refresh_badges()
        self._save_library_state()
        self.statusBar().showMessage(f"Renamed to {new_path.name}")
        self._update_status_bar()

    def _renumber_by_creation_time(self) -> None:
        if not self.library.items:
            return
        if not self._can_navigate_away():
            return

        groups = find_sequence_groups(self.library.items)
        if not groups:
            QMessageBox.information(
                self,
                "Renumber by Creation Time",
                "No groups of same-basename sequenced images (e.g. toto001.jpg, "
                "toto002.jpg) were found in this folder.",
            )
            return

        dialog = RenumberDialog(groups, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.chosen_names:
            return

        current_item = self.library.current_item
        total_renamed = 0
        summary_lines: list[str] = []
        all_errors: list[str] = []

        # Groups are disjoint (a filename matches exactly one basename), so
        # renumbering each one against the same items list in turn is safe —
        # earlier groups' renames can't start matching a later group's name.
        for name in dialog.chosen_names:
            report = renumber_by_creation_time(self.library.items, name)
            total_renamed += report.renamed
            if report.renamed:
                self.library.renamed_names[name] = max(self.library.renamed_names.get(name, 0), report.renamed)
            summary = f"{name}: {report.renamed} renamed"
            if report.errors:
                summary += f", {len(report.errors)} error(s)"
            summary_lines.append(summary)
            all_errors.extend(report.errors)

        if total_renamed:
            # Filenames changed (possibly the sort order too); re-sort in place
            # rather than reloading from disk, so status/rating stay attached
            # to the right ImageItem objects instead of being re-read by name.
            self.library.items.sort(key=lambda item: item.path)
            if current_item is not None:
                self.library.current_index = self.library.items.index(current_item)
            self.edit_session = None
            self.thumbnail_list.set_items(self.library.items)
            self._save_library_state()
            self._show_current()

        message = "\n".join(summary_lines)
        if all_errors:
            message += "\n\nErrors:\n" + "\n".join(all_errors)
        QMessageBox.information(self, "Renumber by Creation Time", message)

    def _rename_by_creation_date(self) -> None:
        if not self.library.items:
            return
        if not self._can_navigate_away():
            return

        confirm = QMessageBox.question(
            self,
            "Rename All by Creation Date",
            f"Rename all {len(self.library.items)} image(s) in this folder to "
            "pYYYYmmdd_hhmmss.ext, based on each photo's creation date?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        current_item = self.library.current_item
        report = rename_by_creation_date(self.library.items)

        if report.renamed:
            # Filenames (and their sort order) changed; re-sort in place rather
            # than reloading from disk, so status/rating stay attached to the
            # right ImageItem objects instead of being re-read by name.
            self.library.items.sort(key=lambda item: item.path)
            if current_item is not None:
                self.library.current_index = self.library.items.index(current_item)
            self.edit_session = None
            self.thumbnail_list.set_items(self.library.items)
            self._save_library_state()
            self._show_current()

        message = f"{report.renamed} renamed"
        if report.errors:
            message += f", {len(report.errors)} error(s):\n" + "\n".join(report.errors)
        QMessageBox.information(self, "Rename All by Creation Date", message)

    # -- Editing ----------------------------------------------------

    def _ensure_edit_session(self) -> EditSession | None:
        item = self.library.current_item
        if item is None:
            return None
        if self.edit_session is None or self.edit_session.source_path != item.path:
            self.edit_session = EditSession.from_path(item.path)
            self.edit_panel.reset_adjustment_sliders()
        return self.edit_session

    def _refresh_preview(self) -> None:
        if self.edit_session is None:
            return
        image = self.edit_session.render()
        self.viewer.set_image(pil_to_qimage(image))
        self.edit_panel.set_image_size(image.width, image.height)
        self.edit_panel.set_history_enabled(self.edit_session.can_undo(), self.edit_session.can_redo())

    def _on_side_tab_changed(self, _index: int) -> None:
        """The side panel is a single tab widget now (Image Information /
        Edit Image / Face Recognition / Search by Name), exactly one visible
        at a time -- redo whatever used to happen when a dock was
        independently toggled on."""
        current = self.side_tabs.currentWidget()
        # Crop mode and face-edit mode are both interpreted by the image
        # viewer's mouse handlers regardless of which tab is showing --
        # the viewer isn't itself part of any tab, so it stays visible and
        # interactive when the user switches away. Without this, leaving
        # either mode on while switching tabs let a drag on the photo
        # silently crop it or silently add a face box the user never meant
        # to draw.
        if current is not self.edit_panel:
            self.viewer.set_crop_mode(False)
            self.edit_panel.set_crop_mode_active(False)
        if RECOGNITION_AVAILABLE and current is not self.face_panel:
            self.viewer.set_face_edit_mode(False)
            self.face_panel.set_edit_mode_active(False)
        if current is self.edit_panel:
            if self._ensure_edit_session() is not None:
                self._refresh_preview()
        elif RECOGNITION_AVAILABLE and current is self.face_panel:
            self._request_face_detection()
        elif RECOGNITION_AVAILABLE and current is self.search_panel:
            self.search_panel.refresh_people()

    # -- Face recognition ----------------------------------------------------

    def _request_face_detection(self) -> None:
        """Kick off (or re-display cached) face detection for the current photo.

        No-op if the Face Recognition tab isn't the active one -- browsing
        without it selected shouldn't pay detection's per-photo cost (a few
        hundred ms on this machine's GPU, more on CPU) for a feature the user
        isn't looking at.
        """
        if self.side_tabs.currentWidget() is not self.face_panel:
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
        # Kept alive until it finishes, same reasoning as _pending_image_workers:
        # otherwise its signals QObject could be garbage-collected mid-run.
        self._pending_face_workers.append(worker)
        worker.signals.finished.connect(
            lambda path, records, error, w=worker: self._on_faces_detected(path, records, error, w)
        )
        # Same elevated priority as the visible full-image load (see
        # IMAGE_LOAD_PRIORITY) -- this is just as much "what the user is
        # looking at right now" as the image itself, and without it,
        # switching to the Face Recognition tab right after opening a large
        # never-before-processed folder could queue behind thousands of
        # pending thumbnail jobs before detection for the current photo
        # even starts.
        self._thread_pool.start(worker, IMAGE_LOAD_PRIORITY)

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

    def _on_face_edit_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            # Crop and face-edit modes both interpret mouse drags on the
            # viewer, and _face_edit_mode wins that check first -- leaving
            # Crop's button checked while it's actually non-functional would
            # be confusing, so turn it off.
            self.viewer.set_crop_mode(False)
            self.edit_panel.set_crop_mode_active(False)
        self.viewer.set_face_edit_mode(enabled)

    def _on_face_box_added(self, box: tuple[int, int, int, int]) -> None:
        item = self.library.current_item
        if item is None:
            return
        record = self.face_catalog.add_manual_face(item.path, box)
        if self._current_face_path == item.path:
            self._current_face_records.append(record)
        self._save_face_catalog()
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
        self._save_face_catalog()
        self._update_face_display()

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

        self._save_person_gallery()
        self._save_face_catalog()
        self._update_face_display()

    def _show_manage_people_dialog(self) -> None:
        dialog = ManagePeopleDialog(self.person_gallery, self)
        dialog.exec()
        # Only the currently-loaded folder's face records can be updated
        # here; a merged-away/forgotten person's label in a folder that isn't
        # open right now will just show as unconfirmed next time that folder
        # is opened (for a merge, its correct suggestion should resurface on
        # its own, since the same embeddings now live under the kept person).
        if dialog.merges or dialog.forgotten_ids:
            for removed_id, kept_id in dialog.merges:
                self.face_catalog.remap_person(removed_id, kept_id)
            for forgotten_id in dialog.forgotten_ids:
                self.face_catalog.forget_person(forgotten_id)
            self._save_face_catalog()
            self._update_face_display()

    def _on_search_photo_chosen(self, path: Path) -> None:
        index = next((i for i, item in enumerate(self.library.items) if item.path == path), None)
        if index is None or not self._can_navigate_away():
            return
        self.library.current_index = index
        self._show_current()

    def _on_forget_all_faces(self) -> None:
        if not self.person_gallery.people:
            return
        confirm = QMessageBox.question(
            self,
            "Forget All Faces",
            "Forget every named person and all their recognition data? This cannot be undone. "
            "Faces already labeled in photos will show as unconfirmed again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.person_gallery.people = []
        self._save_person_gallery()
        self.face_catalog.unassign_all_people()
        self._save_face_catalog()
        self._update_face_display()

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
        if self._current_qimage is None or self._current_qimage_path != self.library.current_item.path:
            return QPixmap()  # image for this photo hasn't finished loading yet; see _on_image_loaded
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

    def _connect_edit_panel(self) -> None:
        panel = self.edit_panel

        panel.rotate_cw.connect(lambda: self._apply_edit(lambda es: es.rotate(clockwise=True)))
        panel.rotate_ccw.connect(lambda: self._apply_edit(lambda es: es.rotate(clockwise=False)))
        panel.flip_horizontal.connect(lambda: self._apply_edit(lambda es: es.flip_horizontal()))
        panel.flip_vertical.connect(lambda: self._apply_edit(lambda es: es.flip_vertical()))

        panel.crop_mode_toggled.connect(self._on_crop_mode_toggled)

        panel.adjustments_changed.connect(self._on_adjustments_changed)
        panel.adjustments_committed.connect(self._on_adjustments_committed)

        panel.resize_requested.connect(
            lambda w, h: self._apply_edit(lambda es: es.resize((w, h)))
        )

        panel.undo_requested.connect(lambda: self._apply_edit(lambda es: es.undo()))
        panel.redo_requested.connect(lambda: self._apply_edit(lambda es: es.redo()))
        panel.reset_requested.connect(self._on_reset_edits)

        panel.save_copy_requested.connect(lambda: self._save_edit(mode="copy"))
        panel.save_as_requested.connect(lambda: self._save_edit(mode="as"))
        panel.save_overwrite_requested.connect(lambda: self._save_edit(mode="overwrite"))

    def _apply_edit(self, fn) -> None:
        session = self._ensure_edit_session()
        if session is None:
            return
        fn(session)
        self._refresh_preview()

    def _on_crop_mode_toggled(self, enabled: bool) -> None:
        if enabled:
            self._ensure_edit_session()
            if RECOGNITION_AVAILABLE:
                # See _on_face_edit_mode_toggled -- same reasoning, other
                # direction.
                self.viewer.set_face_edit_mode(False)
                self.face_panel.set_edit_mode_active(False)
        self.viewer.set_crop_mode(enabled)

    def _on_crop_selected(self, box: tuple[int, int, int, int]) -> None:
        session = self._ensure_edit_session()
        if session is None:
            return
        session.crop(box)
        self.viewer.set_crop_mode(False)
        self.edit_panel.set_crop_mode_active(False)
        self._refresh_preview()

    def _on_adjustments_changed(self, brightness: float, contrast: float, saturation: float) -> None:
        if self._ensure_edit_session() is None:
            return
        # Coalesce bursts of slider ticks (one per pixel of mouse movement) into a
        # bounded render rate, instead of re-rendering the full image on every tick.
        self._pending_adjustments = (brightness, contrast, saturation)
        self._adjustment_timer.start()

    def _apply_pending_adjustments(self) -> None:
        if self._pending_adjustments is None or self.edit_session is None:
            return
        self.edit_session.set_adjustments(*self._pending_adjustments)
        self._pending_adjustments = None
        self._refresh_preview()

    def _on_adjustments_committed(self) -> None:
        if self.edit_session is None:
            return
        if self._adjustment_timer.isActive():
            self._adjustment_timer.stop()
            self._apply_pending_adjustments()
        self.edit_session.commit_adjustments()
        self._refresh_preview()

    def _on_reset_edits(self) -> None:
        if self.edit_session is None:
            return
        self.edit_session.reset()
        self.edit_panel.reset_adjustment_sliders()
        self._refresh_preview()
        self.edit_panel.set_history_enabled(False, False)

    def _save_edit(self, mode: str) -> None:
        session = self.edit_session
        if session is None or not session.has_edits():
            self.edit_panel.set_status_message("No edits to save.")
            return

        try:
            if mode == "copy":
                saved_path = session.save(overwrite=False)
            elif mode == "overwrite":
                confirm = QMessageBox.question(
                    self,
                    "Overwrite Original",
                    f"Overwrite {session.source_path.name} with the edited version? "
                    "This cannot be undone.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return
                saved_path = session.save(overwrite=True)
                if RECOGNITION_AVAILABLE:
                    # The old cached boxes/embeddings were computed against
                    # the pre-edit pixel geometry (rotation/flip/crop) and
                    # are now wrong -- drop them so the next detection
                    # request re-processes the actual current file.
                    self.face_catalog.invalidate(saved_path)
                    if self._current_face_path == saved_path:
                        self._request_face_detection()
            elif mode == "as":
                default_name = str(
                    session.source_path.with_name(
                        f"{session.source_path.stem}_edited{session.source_path.suffix}"
                    )
                )
                chosen, _ = QFileDialog.getSaveFileName(self, "Save As", default_name)
                if not chosen:
                    return
                saved_path = session.save(path=Path(chosen))
            else:
                return
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return

        self.edit_panel.set_status_message(f"Saved to {saved_path}")

    # -- Apply culling ----------------------------------------------------

    def _apply_culling(self) -> None:
        if self.library.folder is None:
            return
        if not self._can_navigate_away():
            return

        counts = self.library.counts()
        if counts["selected"] == 0 and counts["rejected"] == 0:
            QMessageBox.information(self, "Apply Culling", "No images are marked as selected or rejected.")
            return

        dialog = ApplyCullingDialog(counts, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        mode, selected_dir, rejected_dir = dialog.values()

        confirm = QMessageBox.question(
            self,
            "Apply Culling",
            f"{mode.capitalize()} {counts['selected']} selected and {counts['rejected']} rejected "
            f"image(s) into '{selected_dir}/' and '{rejected_dir}/'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        # Snapshot which items are candidates to move, and their pre-move
        # path, so we can tell afterward which ones actually moved (vs.
        # errored partway and stayed put) -- needed below to invalidate just
        # their cached face data, not everything.
        moved_candidates = [
            (item, item.path) for item in self.library.items if item.status in (Status.SELECTED, Status.REJECTED)
        ]

        report = apply_culling(self.library, mode=mode, selected_dir=selected_dir, rejected_dir=rejected_dir)

        if RECOGNITION_AVAILABLE and mode == "move":
            # Drop cached face data for anything that actually left this
            # folder *before* it can be saved -- otherwise the save below
            # would write those now-orphaned entries into this folder's
            # .picsel_faces.json, ready to misattribute results if a future
            # photo reuses the same filename here (the same class of bug as
            # the ratings/status one this method already guards against by
            # saving only after the reload below).
            for item, original_path in moved_candidates:
                if item.path != original_path:
                    self.face_catalog.invalidate(original_path)

        if mode == "copy":
            # Nothing moved out of this folder -- self.library.items still
            # accurately reflects what's here, so this is safe to save now.
            self._save_library_state()

        message = f"Moved {report.moved_selected} selected and {report.moved_rejected} rejected image(s)."
        if report.errors:
            message += "\n\nErrors:\n" + "\n".join(report.errors)
        QMessageBox.information(self, "Apply Culling", message)

        if mode == "move":
            folder = self.library.folder
            previous_name = self.library.current_item.name if self.library.current_item else None
            previous_index = self.library.current_index
            self.edit_session = None
            try:
                self.library.load(folder)
            except OSError as exc:
                QMessageBox.critical(self, "Apply Culling", f"Photos were moved, but reloading {folder} failed:\n{exc}")
                return
            if self.library.load_error:
                QMessageBox.warning(self, "Ratings/Status", self.library.load_error)
            # Only now, after the reload -- self.library.items reflects what's
            # actually still in this folder, so the rewritten state file
            # can't include stale entries for photos that just moved out
            # (save_state() rewrites the whole file from self.items, it
            # doesn't merge, so this naturally scrubs them).
            self._save_library_state()
            if RECOGNITION_AVAILABLE:
                # Reload rather than keep the in-memory cache: some cached
                # entries now refer to photos that just moved to selected/
                # rejected/, and stale entries could otherwise misattribute
                # results if a moved photo's filename gets reused later --
                # already invalidated above, so this save is now clean too.
                self._save_face_catalog()
                self.face_catalog.load(folder)
                if self.face_catalog.load_error:
                    QMessageBox.warning(self, "Face Data", self.face_catalog.load_error)
            if self._sort_mode != "name" and self.library.items:
                self.library.sort_items(key=self._sort_key(self._sort_mode))
            self.thumbnail_list.set_items(self.library.items)
            if self.library.items:
                # The previously current item may itself have been moved out by
                # this culling pass, so it won't be found by name; fall back to
                # the closest valid index to where it used to be.
                match = next(
                    (i for i, item in enumerate(self.library.items) if item.name == previous_name), None
                )
                self.library.current_index = (
                    match if match is not None else min(previous_index, len(self.library.items) - 1)
                )
                self._show_current()
            else:
                self.viewer.set_image(QImage())
                self.metadata_panel.set_image(None)
                self._update_status_bar()

    # -- Lifecycle ----------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_library_state()
        if RECOGNITION_AVAILABLE:
            self._save_face_catalog()
        # Thumbnail and image-load workers run on the shared global thread pool.
        # If any are still running when Qt starts tearing down, they crash trying
        # to emit `finished` on a signals object whose C++ side is already gone.
        # Cancel anything not yet started and block until in-flight tasks finish.
        self._thread_pool.clear()
        self._thread_pool.waitForDone()
        super().closeEvent(event)
