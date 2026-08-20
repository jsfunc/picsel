"""Read-only panel listing all available metadata for the current image."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QSizePolicy, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from tamis.metadata import MetadataSection, extract_metadata


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
        """Synchronous convenience wrapper around `set_sections` -- reads and
        displays `path`'s metadata directly on the calling thread. Prefer
        dispatching a `tamis.thumbnails.MetadataLoadWorker` and calling
        `set_sections` with its result instead when called from the UI
        thread during normal navigation: `extract_metadata` does file I/O
        plus EXIF/GPS IFD parsing, and doing that synchronously on every
        photo navigation broke this app's otherwise-consistent
        "decode/detect off the UI thread" pattern.
        """
        if path is None:
            self.set_sections([])
            return
        self.set_sections(extract_metadata(path))

    def set_sections(self, sections: list[MetadataSection]) -> None:
        self.tree.clear()
        for title, rows in sections:
            section_item = QTreeWidgetItem([title, ""])
            font = section_item.font(0)
            font.setBold(True)
            section_item.setFont(0, font)
            self.tree.addTopLevelItem(section_item)
            for label, value in rows:
                section_item.addChild(QTreeWidgetItem([label, value]))

        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
