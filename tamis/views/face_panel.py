"""Face-recognition panel: detection status, confidence threshold, and a
per-face list for labeling and manual add/dismiss overrides.

Purely presentational, like EditPanel -- emits signals describing the
requested action (threshold change, edit-mode toggle, a name typed for a
face, a face removed) and the owning window applies them to FaceCatalog /
PersonGallery and refreshes the display. Deliberately has no dependency on
`tamis.recognition` types (plain values only), so this module -- and the
views package as a whole -- stays importable even without the optional
recognition deps installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from tamis.recognition.detector import DEFAULT_MIN_CONFIDENCE
from tamis.views.theme import HIGH_SIMILARITY_COLOR, LOW_SIMILARITY_COLOR

SLIDER_RANGE = (0, 100)  # maps to a 0.0 - 1.0 confidence factor
THUMBNAIL_SIZE = 48

# Similarity color scale for suggestions: there's no single threshold that
# cleanly separates "recognized" from "not" (see PersonGallery.identify's
# docstring), so confidence is conveyed continuously by color instead of a
# yes/no cutoff. Below _LOW, solid red; above _HIGH, solid green; linearly
# blended in between.
_LOW_SIMILARITY_COLOR = LOW_SIMILARITY_COLOR
_HIGH_SIMILARITY_COLOR = HIGH_SIMILARITY_COLOR
_LOW_SIMILARITY = 0.3
_HIGH_SIMILARITY = 0.7


def _similarity_color(similarity: float) -> QColor:
    t = (similarity - _LOW_SIMILARITY) / (_HIGH_SIMILARITY - _LOW_SIMILARITY)
    t = max(0.0, min(1.0, t))
    return QColor(
        round(_LOW_SIMILARITY_COLOR.red() + t * (_HIGH_SIMILARITY_COLOR.red() - _LOW_SIMILARITY_COLOR.red())),
        round(_LOW_SIMILARITY_COLOR.green() + t * (_HIGH_SIMILARITY_COLOR.green() - _LOW_SIMILARITY_COLOR.green())),
        round(_LOW_SIMILARITY_COLOR.blue() + t * (_HIGH_SIMILARITY_COLOR.blue() - _LOW_SIMILARITY_COLOR.blue())),
    )


@dataclass
class FaceEntry:
    """One row's worth of display data for a visible face -- built by the
    owning window from a FaceRecord plus whatever PersonGallery suggests."""

    thumbnail: QPixmap
    dropdown_entries: list[tuple[str, float]]  # every known person + similarity (PersonGallery.rank_all), ranked
    top_suggestion: str  # "" unless PersonGallery.identify() found any candidate at all
    top_suggestion_similarity: float
    confirmed_name: str  # "" if this face hasn't been labeled yet
    is_manual: bool  # True for a user-added box (no detector confidence)


class FaceRow(QWidget):
    name_confirmed = Signal(int, str)  # index, name (name may be "" to unassign)
    remove_requested = Signal(int)  # index; dismisses a detector box or deletes a manual one

    def __init__(self, index: int, entry: FaceEntry, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        self._last_committed = entry.confirmed_name
        self._top_suggestion = entry.top_suggestion

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        thumb_label = QLabel()
        thumb_label.setPixmap(entry.thumbnail)
        thumb_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        layout.addWidget(thumb_label)

        # A colored QLabel, not QLineEdit placeholder text: many Qt styles
        # compute their own faded placeholder color and ignore
        # QPalette::PlaceholderText entirely, so that route can't reliably
        # convey confidence via color. A plain QLabel with a stylesheet color
        # is respected everywhere.
        self.suggestion_label = QLabel()
        show_suggestion = bool(entry.top_suggestion) and not entry.confirmed_name
        if show_suggestion:
            self.suggestion_label.setText(f"{entry.top_suggestion} ?")
            color = _similarity_color(entry.top_suggestion_similarity)
            self.suggestion_label.setStyleSheet(f"color: rgb({color.red()}, {color.green()}, {color.blue()});")
        self.suggestion_label.setVisible(show_suggestion)
        layout.addWidget(self.suggestion_label)

        self.name_combo = QComboBox()
        self.name_combo.setEditable(True)
        for name, similarity in entry.dropdown_entries:
            self.name_combo.addItem(name)
            color = _similarity_color(similarity)
            self.name_combo.setItemData(self.name_combo.count() - 1, color, Qt.ItemDataRole.ForegroundRole)
            self.name_combo.setItemData(
                self.name_combo.count() - 1, f"similarity: {similarity:.2f}", Qt.ItemDataRole.ToolTipRole
            )
        # The suggestion is shown separately (suggestion_label above), not as
        # live text here: if it were pre-filled, merely tabbing past this
        # field (never intending to accept it) would "confirm" it on
        # focus-out. Confirming is always a deliberate act -- pick a dropdown
        # item, click Confirm, or type + Enter.
        self.name_combo.setCurrentText(entry.confirmed_name)
        self.name_combo.lineEdit().setPlaceholderText("Name...")
        self.name_combo.lineEdit().editingFinished.connect(self._commit_current_text)
        self.name_combo.textActivated.connect(self._commit_text)
        self._set_bold(bool(entry.confirmed_name))
        layout.addWidget(self.name_combo)

        confirm_button = QPushButton("✓")
        confirm_button.setFixedWidth(28)
        confirm_button.setToolTip("Confirm name (accepts the suggestion above if the field is empty)")
        confirm_button.clicked.connect(self._on_confirm_clicked)
        layout.addWidget(confirm_button)

        remove_button = QPushButton("✕")
        remove_button.setFixedWidth(28)
        remove_button.setToolTip("Remove this face" if entry.is_manual else "Not a face")
        remove_button.clicked.connect(lambda: self.remove_requested.emit(self._index))
        layout.addWidget(remove_button)

    def _commit_current_text(self) -> None:
        self._commit_text(self.name_combo.currentText())

    def _on_confirm_clicked(self) -> None:
        text = self.name_combo.currentText().strip()
        if not text and self._top_suggestion:
            # Nothing typed yet, but there's a suggestion showing as a
            # placeholder hint -- clicking Confirm with an empty field means
            # "yes, that one", so accept it without requiring the user to
            # open the dropdown or retype it themselves.
            text = self._top_suggestion
            self.name_combo.setCurrentText(text)
        self._commit_text(text)

    def _commit_text(self, text: str) -> None:
        name = text.strip()
        if name == self._last_committed:
            return  # unchanged -- avoid re-adding a duplicate embedding sample on every focus-out
        self._last_committed = name
        self._set_bold(bool(name))  # a name just committed here is validated, not merely suggested
        self.suggestion_label.setVisible(not name)  # redundant with the now-confirmed text otherwise
        self.name_confirmed.emit(self._index, name)

    def _set_bold(self, bold: bool) -> None:
        font = self.name_combo.lineEdit().font()
        font.setBold(bold)
        self.name_combo.lineEdit().setFont(font)


class FacePanel(QWidget):
    threshold_changed = Signal(float)
    edit_mode_toggled = Signal(bool)
    name_confirmed = Signal(int, str)  # index into the list last passed to set_faces
    remove_requested = Signal(int)  # index into the list last passed to set_faces
    manage_people_requested = Signal()
    forget_all_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._status_label = QLabel("No photo open.")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._threshold_label = QLabel()
        layout.addWidget(self._threshold_label)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(*SLIDER_RANGE)
        self.threshold_slider.setValue(round(DEFAULT_MIN_CONFIDENCE * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider_changed)
        layout.addWidget(self.threshold_slider)

        self.edit_button = QPushButton("Edit Faces")
        self.edit_button.setCheckable(True)
        self.edit_button.setToolTip(
            "While enabled: click a box to remove it, or drag on the photo to add a missed face."
        )
        self.edit_button.toggled.connect(self.edit_mode_toggled)
        layout.addWidget(self.edit_button)

        self.faces_list = QListWidget()
        self.faces_list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.faces_list)

        gallery_row = QHBoxLayout()
        self.manage_people_button = QPushButton("Manage People...")
        self.manage_people_button.setToolTip("Rename or merge known people (e.g. duplicates from different capitalization).")
        self.manage_people_button.clicked.connect(self.manage_people_requested)
        gallery_row.addWidget(self.manage_people_button)

        self.forget_all_button = QPushButton("Forget All Faces")
        self.forget_all_button.setToolTip("Erase every named person and their recognition data. Cannot be undone.")
        self.forget_all_button.clicked.connect(self.forget_all_requested)
        gallery_row.addWidget(self.forget_all_button)
        layout.addLayout(gallery_row)

        self._update_threshold_label(self.threshold_slider.value())

    def _on_threshold_slider_changed(self, value: int) -> None:
        self._update_threshold_label(value)
        self.threshold_changed.emit(self.threshold())

    def _update_threshold_label(self, value: int) -> None:
        self._threshold_label.setText(f"Confidence threshold: {value / 100.0:.2f}")

    def threshold(self) -> float:
        return self.threshold_slider.value() / 100.0

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_edit_mode_active(self, active: bool) -> None:
        self.edit_button.setChecked(active)

    def set_faces(self, entries: list[FaceEntry]) -> None:
        self.faces_list.clear()
        rows: list[FaceRow] = []
        for index, entry in enumerate(entries):
            row = FaceRow(index, entry)
            row.name_confirmed.connect(self.name_confirmed)
            row.remove_requested.connect(self.remove_requested)
            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self.faces_list.addItem(item)
            self.faces_list.setItemWidget(item, row)
            rows.append(row)
        # Tab from one face's name field moves to the next face's (Shift+Tab
        # goes back) -- QListWidget's own tab handling doesn't chain focus
        # between embedded item widgets on its own.
        for previous_row, next_row in zip(rows, rows[1:]):
            QWidget.setTabOrder(previous_row.name_combo, next_row.name_combo)
