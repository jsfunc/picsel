"""Persistent "Search by Name" tab: scans every photo in the currently open
folder for a named person.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from picsel.recognition.worker import FolderSearchWorker


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

    def cancel_if_targeting(self, person_ids: set) -> None:
        """Cancel the in-flight search, if any, if it's looking for one of
        the given person ids -- called when a merge/forget in Manage People
        removes a Person this scan holds a direct reference to (searching
        would otherwise keep running against a Person no longer in the
        gallery, silently going quiet instead of erroring)."""
        if self._worker is not None and self._worker.person.id in person_ids:
            self._on_cancel_clicked()

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

    def _on_search_finished(self, error: str, failed_paths: list) -> None:
        self._worker = None
        self.search_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if error:
            self.status_label.setText(f"Search failed: {error}")
            return
        confirmed_count = sum(1 for hit in self._hits if hit.confirmed)
        prefix = "Cancelled. " if self._cancel_requested else ""
        suffix = (
            f" ({len(failed_paths)} photo(s) could not be processed and were skipped.)" if failed_paths else ""
        )
        self.status_label.setText(
            f"{prefix}{len(self._hits)} occurrence(s) found ({confirmed_count} confirmed).{suffix}"
        )

    def _on_item_clicked(self, list_item: QListWidgetItem) -> None:
        path = list_item.data(Qt.ItemDataRole.UserRole)
        if path is not None:
            self.photo_chosen.emit(path)
