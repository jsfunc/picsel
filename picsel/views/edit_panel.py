"""Editing controls: transform, adjustments, resize, history and save actions.

This widget is purely presentational — it emits signals describing the
requested action and the owning window (MainWindow) applies them to the
current EditSession and refreshes the preview.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

SLIDER_RANGE = (0, 200)  # maps to factor 0.0 - 2.0
SLIDER_DEFAULT = 100  # factor 1.0


class EditPanel(QWidget):
    rotate_cw = Signal()
    rotate_ccw = Signal()
    flip_horizontal = Signal()
    flip_vertical = Signal()
    crop_mode_toggled = Signal(bool)
    adjustments_changed = Signal(float, float, float)  # brightness, contrast, saturation
    adjustments_committed = Signal()
    resize_requested = Signal(int, int)
    undo_requested = Signal()
    redo_requested = Signal()
    reset_requested = Signal()
    save_copy_requested = Signal()
    save_overwrite_requested = Signal()
    save_as_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image_size = (0, 0)
        self._lock_aspect = True

        # Commits an adjustment to undo history after a pause in slider
        # activity, regardless of input method -- sliderReleased (below)
        # only fires after a mouse-drag release, so arrow keys/Page Up-Down/
        # Home-End (fully valid QSlider interactions, and the only path for
        # keyboard-only or assistive-tech use) previously updated the live
        # preview but never landed in undo history. commit_adjustments() is
        # idempotent (a no-op once already committed), so this firing after
        # sliderReleased already committed is harmless.
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.setInterval(500)
        self._commit_timer.timeout.connect(self.adjustments_committed)

        layout = QVBoxLayout(self)

        layout.addWidget(self._build_transform_group())
        layout.addWidget(self._build_adjustments_group())
        layout.addWidget(self._build_resize_group())
        layout.addWidget(self._build_history_group())
        layout.addWidget(self._build_save_group())
        layout.addStretch(1)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    # -- Transform --------------------------------------------------

    def _build_transform_group(self) -> QGroupBox:
        group = QGroupBox("Transform")
        layout = QHBoxLayout(group)

        rotate_ccw_btn = QPushButton("Rotate ⟲")
        rotate_ccw_btn.clicked.connect(self.rotate_ccw)
        rotate_cw_btn = QPushButton("Rotate ⟳")
        rotate_cw_btn.clicked.connect(self.rotate_cw)
        flip_h_btn = QPushButton("Flip ↔")
        flip_h_btn.clicked.connect(self.flip_horizontal)
        flip_v_btn = QPushButton("Flip ↕")
        flip_v_btn.clicked.connect(self.flip_vertical)

        self.crop_button = QPushButton("Crop")
        self.crop_button.setCheckable(True)
        self.crop_button.toggled.connect(self.crop_mode_toggled)

        for btn in (rotate_ccw_btn, rotate_cw_btn, flip_h_btn, flip_v_btn, self.crop_button):
            layout.addWidget(btn)

        return group

    def set_crop_mode_active(self, active: bool) -> None:
        self.crop_button.setChecked(active)

    # -- Adjustments --------------------------------------------------

    def _build_adjustments_group(self) -> QGroupBox:
        group = QGroupBox("Adjustments")
        layout = QFormLayout(group)

        self.brightness_slider = self._make_slider()
        self.contrast_slider = self._make_slider()
        self.saturation_slider = self._make_slider()

        layout.addRow("Brightness", self.brightness_slider)
        layout.addRow("Contrast", self.contrast_slider)
        layout.addRow("Saturation", self.saturation_slider)

        return group

    def _make_slider(self) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(*SLIDER_RANGE)
        slider.setValue(SLIDER_DEFAULT)
        slider.valueChanged.connect(self._emit_adjustments)
        slider.sliderReleased.connect(self._on_slider_released)
        return slider

    def _on_slider_released(self) -> None:
        self._commit_timer.stop()  # already committing now, no need for the fallback timer to fire too
        self.adjustments_committed.emit()

    def _emit_adjustments(self) -> None:
        self.adjustments_changed.emit(
            self.brightness_slider.value() / 100.0,
            self.contrast_slider.value() / 100.0,
            self.saturation_slider.value() / 100.0,
        )
        self._commit_timer.start()

    def reset_adjustment_sliders(self) -> None:
        for slider in (self.brightness_slider, self.contrast_slider, self.saturation_slider):
            slider.blockSignals(True)
            slider.setValue(SLIDER_DEFAULT)
            slider.blockSignals(False)

    # -- Resize --------------------------------------------------

    def _build_resize_group(self) -> QGroupBox:
        group = QGroupBox("Resize")
        layout = QFormLayout(group)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.width_spin.valueChanged.connect(self._on_width_changed)
        self.height_spin.valueChanged.connect(self._on_height_changed)

        self.lock_aspect_checkbox = QCheckBox("Lock aspect ratio")
        self.lock_aspect_checkbox.setChecked(True)
        self.lock_aspect_checkbox.toggled.connect(self._on_lock_aspect_toggled)

        apply_btn = QPushButton("Apply Resize")
        apply_btn.clicked.connect(
            lambda: self.resize_requested.emit(self.width_spin.value(), self.height_spin.value())
        )

        layout.addRow("Width", self.width_spin)
        layout.addRow("Height", self.height_spin)
        layout.addRow(self.lock_aspect_checkbox)
        layout.addRow(apply_btn)

        return group

    def _on_lock_aspect_toggled(self, checked: bool) -> None:
        self._lock_aspect = checked

    def _on_width_changed(self, width: int) -> None:
        if not self._lock_aspect or self._image_size[0] == 0:
            return
        ratio = self._image_size[1] / self._image_size[0]
        self.height_spin.blockSignals(True)
        self.height_spin.setValue(max(1, round(width * ratio)))
        self.height_spin.blockSignals(False)

    def _on_height_changed(self, height: int) -> None:
        if not self._lock_aspect or self._image_size[1] == 0:
            return
        ratio = self._image_size[0] / self._image_size[1]
        self.width_spin.blockSignals(True)
        self.width_spin.setValue(max(1, round(height * ratio)))
        self.width_spin.blockSignals(False)

    def set_image_size(self, width: int, height: int) -> None:
        self._image_size = (width, height)
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)

    # -- History --------------------------------------------------

    def _build_history_group(self) -> QGroupBox:
        group = QGroupBox("History")
        layout = QHBoxLayout(group)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_requested)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo_requested)
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_requested)

        for btn in (self.undo_button, self.redo_button, reset_btn):
            layout.addWidget(btn)

        return group

    def set_history_enabled(self, can_undo: bool, can_redo: bool) -> None:
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)

    # -- Save --------------------------------------------------

    def _build_save_group(self) -> QGroupBox:
        group = QGroupBox("Save")
        layout = QVBoxLayout(group)

        save_copy_btn = QPushButton("Save Copy (name_edited.ext)")
        save_copy_btn.clicked.connect(self.save_copy_requested)
        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self.save_as_requested)
        overwrite_btn = QPushButton("Overwrite Original")
        overwrite_btn.clicked.connect(self.save_overwrite_requested)

        for btn in (save_copy_btn, save_as_btn, overwrite_btn):
            layout.addWidget(btn)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        return group

    def set_status_message(self, message: str) -> None:
        self._status_label.setText(message)
