"""Dialog for renaming, merging, forgetting, exporting, and importing people
in a PersonGallery. Only depends on the gallery object it's given, not on
MainWindow or any recognition-worker machinery.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


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

    def _save_gallery(self) -> None:
        try:
            self.gallery.save()
        except OSError as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not save the people gallery:\n{exc}")

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
        self._save_gallery()
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
        self._save_gallery()
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
        self._save_gallery()
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
        self._save_gallery()
        self._refresh_list()
        QMessageBox.information(
            self,
            "Import Gallery",
            f"Imported {added} new people. People already known by the same name had "
            "the imported samples added to their existing entry instead of duplicating them.",
        )
