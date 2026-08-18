"""Main application window: layout, menus, shortcuts, and controller logic."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
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
  E               Toggle edit panel
  M               Toggle metadata panel
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

        # Metadata sits beside the image only (not the full window height like a
        # dock would), so the thumbnail strip below spans the full width.
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.viewer)
        top_splitter.addWidget(self.metadata_panel)
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

        self.edit_dock = QDockWidget("Edit", self)
        self.edit_dock.setWidget(self.edit_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.edit_dock)
        self.edit_dock.setVisible(False)

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

        add("E", self._toggle_edit_panel)
        add("M", self._toggle_metadata_panel)
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

        self.edit_session = None
        try:
            self.library.load(folder)
        except OSError as exc:
            QMessageBox.critical(self, "Open Folder Failed", f"Could not read {folder}:\n{exc}")
            return

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
        self.viewer.set_image(qimage)
        self.edit_panel.set_image_size(qimage.width(), qimage.height())
        self.edit_panel.reset_adjustment_sliders()
        self.edit_panel.set_history_enabled(False, False)
        self.viewer.set_crop_mode(False)
        self.edit_panel.set_crop_mode_active(False)

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

    def _toggle_edit_panel(self) -> None:
        visible = not self.edit_dock.isVisible()
        self.edit_dock.setVisible(visible)
        if visible:
            if self._ensure_edit_session() is not None:
                self._refresh_preview()

    def _toggle_metadata_panel(self) -> None:
        self.metadata_panel.setVisible(not self.metadata_panel.isVisible())

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
            self.library.load(folder)
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
        # Thumbnail and image-load workers run on the shared global thread pool.
        # If any are still running when Qt starts tearing down, they crash trying
        # to emit `finished` on a signals object whose C++ side is already gone.
        # Cancel anything not yet started and block until in-flight tasks finish.
        self._thread_pool.clear()
        self._thread_pool.waitForDone()
        super().closeEvent(event)
