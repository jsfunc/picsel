"""Main application window: layout, menus, shortcuts, and controller logic."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QRect, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QFontDatabase, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from picsel.editing import EditSession
from picsel.io_ops import (
    apply_culling,
    capture_time,
    find_sequence_groups,
    rename_by_creation_date,
    renumber_by_creation_time,
    rename_with_sequence,
)
from picsel.models import ImageItem, ImageLibrary, Status
from picsel.thumbnails import ImageLoadWorker, pil_to_qimage
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
    from picsel.recognition.worker import FaceDetectionWorker, FolderSearchWorker
    from picsel.views.face_panel import THUMBNAIL_SIZE, FaceEntry, FacePanel

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


class ApplyCullingDialog(QDialog):
    def __init__(self, counts: dict[str, int], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Apply Culling")

        layout = QVBoxLayout(self)

        info = QLabel(
            f"Selected: {counts['selected']}   "
            f"Rejected: {counts['rejected']}   "
            f"Unrated (left in place): {counts['unrated']}"
        )
        layout.addWidget(info)

        form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Move", "Copy"])
        self.selected_dir_edit = QLineEdit("selected")
        self.rejected_dir_edit = QLineEdit("rejected")
        form.addRow("Mode", self.mode_combo)
        form.addRow("Selected folder name", self.selected_dir_edit)
        form.addRow("Rejected folder name", self.rejected_dir_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str, str]:
        mode = "move" if self.mode_combo.currentText() == "Move" else "copy"
        selected_dir = self.selected_dir_edit.text().strip() or "selected"
        rejected_dir = self.rejected_dir_edit.text().strip() or "rejected"
        return mode, selected_dir, rejected_dir


class RenameDialog(QDialog):
    """Pick an existing name (click to assign the next number) or add a new one."""

    def __init__(self, names: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rename Image")
        self.chosen_name: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Click a name to rename with the next number, or add a new one:"))

        self.list_widget = QListWidget()
        self.list_widget.addItems(sorted(names, key=str.lower))
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.list_widget.itemClicked.connect(self._on_existing_clicked)
        self.list_widget.itemActivated.connect(self._on_existing_clicked)
        layout.addWidget(self.list_widget)

        add_row = QHBoxLayout()
        self.new_name_edit = QLineEdit()
        self.new_name_edit.setPlaceholderText("New name...")
        self.new_name_edit.returnPressed.connect(self._on_add_new)
        add_button = QPushButton("Add")
        add_button.setAutoDefault(False)
        add_button.clicked.connect(self._on_add_new)
        add_row.addWidget(self.new_name_edit)
        add_row.addWidget(add_button)
        layout.addLayout(add_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_existing_clicked(self, list_item) -> None:
        self.chosen_name = list_item.text()
        self.accept()

    def _on_add_new(self) -> None:
        name = self.new_name_edit.text().strip()
        if not name or any(c in name for c in r'\/:*?"<>|'):
            QMessageBox.warning(self, "Invalid Name", "Enter a name without path separators or /\\:*?\"<>| characters.")
            return
        self.chosen_name = name
        self.accept()


class RenumberDialog(QDialog):
    """Pick one or more detected `<name><digits><ext>` groups to renumber by
    capture time. Each group is renumbered independently, keeping its own
    basename."""

    def __init__(self, groups: dict[str, list[ImageItem]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Renumber by Creation Time")
        self.chosen_names: list[str] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Pick one or more groups to renumber so filenames match capture-time\n"
            "order (e.g. if toto001.jpg was taken after toto002.jpg, they'll\n"
            "swap). Each group keeps its own name. Ctrl/Shift-click to select\n"
            "multiple groups."
        ))

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for name in sorted(groups.keys(), key=str.lower):
            list_item = QListWidgetItem(f"{name}  ({len(groups[name])} images)")
            list_item.setData(Qt.ItemDataRole.UserRole, name)
            self.list_widget.addItem(list_item)
        self.list_widget.itemActivated.connect(self._accept_current_selection)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_current_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_current_selection(self) -> None:
        self.chosen_names = [
            list_item.data(Qt.ItemDataRole.UserRole) for list_item in self.list_widget.selectedItems()
        ]
        if self.chosen_names:
            self.accept()


class ManagePeopleDialog(QDialog):
    """Lists everyone in the PersonGallery; selecting two or more and merging
    folds their embedding samples into one entry (e.g. to fix a duplicate
    created by entering the same person's name with different capitalization).
    """

    def __init__(self, gallery, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage People")
        self.gallery = gallery
        # (removed_person_id, kept_person_id) for each merge done this session,
        # so the caller can update any per-folder face records pointing at a
        # now-gone person id.
        self.merges: list[tuple[str, str]] = []
        # Person ids removed via Forget Name this session, so the caller can
        # clear that label from any per-folder face records pointing at them.
        self.forgotten_ids: list[str] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Select one person to rename, or Ctrl/Shift-click two or more\n"
            "and Merge Selected to combine them (e.g. \"papa\" and \"Papa\")."
        ))

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._refresh_list()
        layout.addWidget(self.list_widget)

        actions_row = QHBoxLayout()
        rename_button = QPushButton("Rename Selected")
        rename_button.clicked.connect(self._on_rename_clicked)
        actions_row.addWidget(rename_button)

        merge_button = QPushButton("Merge Selected")
        merge_button.clicked.connect(self._on_merge_clicked)
        actions_row.addWidget(merge_button)

        forget_button = QPushButton("Forget Name")
        forget_button.setToolTip("Remove the selected people entirely, including all their reference samples.")
        forget_button.clicked.connect(self._on_forget_clicked)
        actions_row.addWidget(forget_button)
        layout.addLayout(actions_row)

        transfer_row = QHBoxLayout()
        export_button = QPushButton("Export Gallery...")
        export_button.clicked.connect(self._on_export_clicked)
        transfer_row.addWidget(export_button)

        import_button = QPushButton("Import Gallery...")
        import_button.clicked.connect(self._on_import_clicked)
        transfer_row.addWidget(import_button)
        layout.addLayout(transfer_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for person in sorted(self.gallery.people, key=lambda p: p.name.lower()):
            sample_word = "sample" if len(person.embeddings) == 1 else "samples"
            list_item = QListWidgetItem(f"{person.name}  ({len(person.embeddings)} {sample_word})")
            list_item.setData(Qt.ItemDataRole.UserRole, person.id)
            self.list_widget.addItem(list_item)

    def _on_rename_clicked(self) -> None:
        selected_ids = [item.data(Qt.ItemDataRole.UserRole) for item in self.list_widget.selectedItems()]
        if len(selected_ids) != 1:
            QMessageBox.information(self, "Rename Person", "Select exactly one person to rename.")
            return

        person = self.gallery.find_by_id(selected_ids[0])
        name, ok = QInputDialog.getText(self, "Rename Person", "New name:", text=person.name)
        if not ok or not name.strip():
            return

        person.name = name.strip()
        self.gallery.save()
        self._refresh_list()

    def _on_merge_clicked(self) -> None:
        selected_ids = [item.data(Qt.ItemDataRole.UserRole) for item in self.list_widget.selectedItems()]
        if len(selected_ids) < 2:
            QMessageBox.information(self, "Merge People", "Select two or more people to merge.")
            return

        default_name = self.gallery.find_by_id(selected_ids[0]).name
        name, ok = QInputDialog.getText(self, "Merge People", "Name for the merged person:", text=default_name)
        if not ok or not name.strip():
            return

        keep_id = selected_ids[0]
        for remove_id in selected_ids[1:]:
            self.gallery.merge(keep_id=keep_id, remove_id=remove_id)
            self.merges.append((remove_id, keep_id))
        self.gallery.find_by_id(keep_id).name = name.strip()
        self.gallery.save()
        self._refresh_list()

    def _on_forget_clicked(self) -> None:
        selected_ids = [item.data(Qt.ItemDataRole.UserRole) for item in self.list_widget.selectedItems()]
        if not selected_ids:
            QMessageBox.information(self, "Forget Name", "Select one or more people to forget.")
            return

        names = [self.gallery.find_by_id(person_id).name for person_id in selected_ids]
        confirm = QMessageBox.question(
            self,
            "Forget Name",
            f"Forget {', '.join(names)} and all their reference samples? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        for person_id in selected_ids:
            self.gallery.remove_person(person_id)
            self.forgotten_ids.append(person_id)
        self.gallery.save()
        self._refresh_list()

    def _on_export_clicked(self) -> None:
        if not self.gallery.people:
            QMessageBox.information(self, "Export Gallery", "There's no one in the gallery yet to export.")
            return
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export Gallery", "picsel_people.json.gz", "Compressed gallery (*.json.gz)"
        )
        if not chosen:
            return
        try:
            self.gallery.export_to(Path(chosen))
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        QMessageBox.information(
            self, "Export Gallery", f"Exported {len(self.gallery.people)} people to {chosen}."
        )

    def _on_import_clicked(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Import Gallery", "", "Gallery files (*.json.gz *.json);;All Files (*)"
        )
        if not chosen:
            return
        try:
            added = self.gallery.import_from(Path(chosen))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            QMessageBox.critical(self, "Import Failed", f"Could not read {chosen}:\n{exc}")
            return
        self.gallery.save()
        self._refresh_list()
        QMessageBox.information(
            self,
            "Import Gallery",
            f"Imported {added} new people. People already known by the same name had "
            "the imported samples added to their existing entry instead of duplicating them.",
        )


class SearchPanel(QWidget):
    """Scans every photo in the currently open folder for a named person.
    Lives as a persistent "Search by Name" tab (not a one-shot dialog), so
    results stay put after clicking through them. Confirmed occurrences
    (already labeled) list first; unconfirmed-but-similar occurrences follow,
    ranked by similarity, down to a tunable cutoff. A single click on a
    result emits `photo_chosen` with its path -- the panel itself doesn't
    know how to navigate the library, that's the owning window's job.
    """

    photo_chosen = Signal(Path)

    def __init__(self, library, face_catalog, person_gallery, thread_pool, get_min_confidence, parent=None) -> None:
        super().__init__(parent)
        self.library = library
        self.face_catalog = face_catalog
        self.person_gallery = person_gallery
        self.thread_pool = thread_pool
        self.get_min_confidence = get_min_confidence
        self._worker = None
        self._hits: list = []
        self._total_photos = 0
        self._cancel_requested = False

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.name_combo = QComboBox()
        self.name_combo.setEditable(True)
        form.addRow("Name", self.name_combo)
        layout.addLayout(form)

        self._similarity_label = QLabel()
        layout.addWidget(self._similarity_label)
        self.similarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_slider.setRange(0, 100)
        self.similarity_slider.setValue(50)
        self.similarity_slider.valueChanged.connect(self._update_similarity_label)
        self._update_similarity_label(self.similarity_slider.value())
        layout.addWidget(self.similarity_slider)

        search_row = QHBoxLayout()
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._on_search_clicked)
        search_row.addWidget(self.search_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        search_row.addWidget(self.cancel_button)
        layout.addLayout(search_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.results_list = QListWidget()
        # Monospace + fixed-width prefix (see _render_results) so filenames
        # line up in a column regardless of whether a row is prefixed by the
        # "✓" mark or a variable-width percentage like "86%".
        self.results_list.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.results_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.results_list)

        self.refresh_people()

    def refresh_people(self) -> None:
        """Repopulate the name dropdown from the gallery's current people,
        keeping whatever's currently typed/selected. Called when this tab
        becomes active, since labeling elsewhere can add new people."""
        current = self.name_combo.currentText()
        self.name_combo.clear()
        self.name_combo.addItems(sorted((p.name for p in self.person_gallery.people), key=str.lower))
        self.name_combo.setCurrentText(current)

    def _update_similarity_label(self, value: int) -> None:
        self._similarity_label.setText(f"Minimum similarity for unconfirmed matches: {value / 100.0:.2f}")

    def _on_search_clicked(self) -> None:
        if self._worker is not None:
            return  # a search is already running
        name = self.name_combo.currentText().strip()
        if not name:
            return
        person = self.person_gallery.find_by_name(name)
        if person is None:
            QMessageBox.information(self, "Search by Name", f'No one named "{name}" in the gallery yet.')
            return
        if not self.library.items:
            QMessageBox.information(self, "Search by Name", "No photos are open to search.")
            return

        self.results_list.clear()
        self._hits = []
        self._total_photos = len(self.library.items)
        self._cancel_requested = False
        self.status_label.setText(self._progress_text(done=0))
        self.search_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        paths = [item.path for item in self.library.items]
        worker = FolderSearchWorker(
            self.face_catalog,
            self.person_gallery,
            person,
            paths,
            min_similarity=self.similarity_slider.value() / 100.0,
            min_confidence=self.get_min_confidence(),
        )
        self._worker = worker  # keep alive until it finishes (same reasoning as other workers)
        worker.signals.photo_processed.connect(self._on_photo_processed)
        worker.signals.finished.connect(self._on_search_finished)
        self.thread_pool.start(worker)

    def _on_cancel_clicked(self) -> None:
        if self._worker is None:
            return
        self._cancel_requested = True
        self._worker.cancel()
        self.cancel_button.setEnabled(False)  # takes effect after the in-flight photo finishes, not instantly

    def _progress_text(self, done: int) -> str:
        confirmed_count = sum(1 for hit in self._hits if hit.confirmed)
        return (
            f"Searching: {done}/{self._total_photos} photos processed, "
            f"{len(self._hits)} found ({confirmed_count} confirmed)."
        )

    def _on_photo_processed(self, hits: list, done: int, total: int) -> None:
        if hits:
            self._hits.extend(hits)
            self._render_results()
        self.status_label.setText(self._progress_text(done))

    def _render_results(self) -> None:
        # Rebuilt on every new hit rather than appended to, so the list stays
        # correctly ordered (confirmed first, then unconfirmed by similarity)
        # throughout the scan, not just once it finishes.
        self.results_list.clear()
        confirmed = [hit for hit in self._hits if hit.confirmed]
        unconfirmed = sorted((hit for hit in self._hits if not hit.confirmed), key=lambda hit: hit.similarity, reverse=True)
        for hit in confirmed + unconfirmed:
            # Right-justified to a fixed width ("100%" is the longest
            # possible prefix) so filenames line up in a column regardless
            # of whether a row shows "✓" or a variable-width percentage --
            # relies on results_list's monospace font to actually line up.
            prefix = "✓ " if hit.confirmed else f"{hit.similarity:.0%}"
            list_item = QListWidgetItem(f"{prefix:>4}  {hit.path.name}")
            list_item.setData(Qt.ItemDataRole.UserRole, hit.path)
            self.results_list.addItem(list_item)

    def _on_search_finished(self, error: str) -> None:
        self._worker = None
        self.search_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if error:
            self.status_label.setText(f"Search failed: {error}")
            return
        confirmed_count = sum(1 for hit in self._hits if hit.confirmed)
        prefix = "Cancelled. " if self._cancel_requested else ""
        self.status_label.setText(f"{prefix}{len(self._hits)} occurrence(s) found ({confirmed_count} confirmed).")

    def _on_item_clicked(self, list_item: QListWidgetItem) -> None:
        path = list_item.data(Qt.ItemDataRole.UserRole)
        if path is not None:
            self.photo_chosen.emit(path)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("picSel")
        self.resize(1280, 860)

        self.library = ImageLibrary()
        self.edit_session: EditSession | None = None
        self._thread_pool = QThreadPool.globalInstance()
        self._sort_mode = "name"  # persists across folder switches within the session

        self._image_load_generation = 0
        self._pending_image_workers: list[ImageLoadWorker] = []

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

    # -- Folder / library ----------------------------------------------------

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self.open_folder(Path(folder))

    def open_folder(self, folder: Path) -> None:
        if not self._can_navigate_away():
            return
        if self.library.folder is not None:
            self.library.save_state()
        if RECOGNITION_AVAILABLE and self.face_catalog.folder is not None:
            self.face_catalog.save()

        self.edit_session = None
        try:
            self.library.load(folder)
        except OSError as exc:
            QMessageBox.critical(self, "Open Folder Failed", f"Could not read {folder}:\n{exc}")
            return
        if RECOGNITION_AVAILABLE:
            self.face_catalog.load(folder)
            if self.face_catalog.load_error:
                QMessageBox.warning(self, "Face Data", self.face_catalog.load_error)

        if self._sort_mode != "name" and self.library.items:
            self.library.sort_items(key=self._sort_key(self._sort_mode))

        self.thumbnail_list.set_items(self.library.items)
        self.setWindowTitle(f"picSel — {folder}")

        if self.library.items:
            self.library.current_index = 0
            self._show_current()
        else:
            self.viewer.set_image(QImage())
            self.metadata_panel.set_image(None)
            self.statusBar().showMessage(f"No supported images found in {folder}")

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
        # edits, and doesn't need to wait on the (async) full-resolution decode.
        self.metadata_panel.set_image(item.path)

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
        self.library.save_state()
        self._update_status_bar()

    def _set_rating(self, rating: int) -> None:
        if self.library.current_item is None:
            return
        self.library.set_rating(self.library.current_index, rating)
        self.thumbnail_list.refresh_badges()
        self.library.save_state()
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
        self.library.save_state()
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
            self.library.save_state()
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
            self.library.save_state()
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
        self._thread_pool.start(worker)

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
        # FaceCatalog or re-runs the model, so it's safe on every slider tick.
        self._update_face_display()

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
        self.face_catalog.save()
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
        self.face_catalog.save()
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

        self.person_gallery.save()
        self.face_catalog.save()
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
            self.face_catalog.save()
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
        self.person_gallery.save()
        self.face_catalog.unassign_all_people()
        self.face_catalog.save()
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

        report = apply_culling(self.library, mode=mode, selected_dir=selected_dir, rejected_dir=rejected_dir)
        self.library.save_state()

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
            if RECOGNITION_AVAILABLE:
                # Reload rather than keep the in-memory cache: some cached
                # entries now refer to photos that just moved to selected/
                # rejected/, and stale entries could otherwise misattribute
                # results if a moved photo's filename gets reused later.
                self.face_catalog.save()
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
        self.library.save_state()
        if RECOGNITION_AVAILABLE:
            self.face_catalog.save()
        # Thumbnail and image-load workers run on the shared global thread pool.
        # If any are still running when Qt starts tearing down, they crash trying
        # to emit `finished` on a signals object whose C++ side is already gone.
        # Cancel anything not yet started and block until in-flight tasks finish.
        self._thread_pool.clear()
        self._thread_pool.waitForDone()
        super().closeEvent(event)
