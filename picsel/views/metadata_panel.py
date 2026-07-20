"""Read-only panel listing all available metadata for the current image."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QSizePolicy, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from picsel.metadata import extract_metadata


class MetadataPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Field", "Value"])
        self.tree.setColumnCount(2)
        self.tree.header().setStretchLastSection(True)
        self.tree.setWordWrap(True)
        self.tree.setUniformRowHeights(True)
        layout.addWidget(self.tree)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        # Wide enough that common values (dates, GPS coordinates, full folder
        # paths) aren't immediately truncated in the dock's default size.
        self.setMinimumWidth(320)

    def set_image(self, path: Path | None) -> None:
        self.tree.clear()
        if path is None:
            return

        for title, rows in extract_metadata(path):
            section_item = QTreeWidgetItem([title, ""])
            font = section_item.font(0)
            font.setBold(True)
            section_item.setFont(0, font)
            self.tree.addTopLevelItem(section_item)
            for label, value in rows:
                section_item.addChild(QTreeWidgetItem([label, value]))

        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
