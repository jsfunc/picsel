"""Small, self-contained modal dialogs used from MainWindow's File menu.

These take only plain data as constructor arguments (never a MainWindow
reference) and report their result via an attribute read after `.exec()`,
so they have no dependency on the rest of the app beyond `tamis.models`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from tamis.models import ImageItem


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
