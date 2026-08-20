"""Owns the current photo's in-memory edit session (rotate/flip/crop/resize/
adjustments/undo-redo) and everything that mutates it -- wires EditPanel's
signals to an EditSession, keeping ImageViewer's preview and EditPanel's
history/status widgets in sync.

Deliberately knows nothing about face recognition: the one place the two
features interact (Crop mode and Edit Faces mode can't both be active, since
both interpret mouse drags on the shared ImageViewer; overwriting a photo
invalidates its cached face data) is resolved by MainWindow, which owns both
controllers and is the natural place for cross-feature glue to live.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFileDialog, QMessageBox

from tamis.editing import EditSession
from tamis.thumbnails import pil_to_qimage


class EditController:
    def __init__(self, parent_widget, library, viewer, edit_panel) -> None:
        self.parent_widget = parent_widget  # for QMessageBox/QFileDialog parenting
        self.library = library
        self.viewer = viewer
        self.edit_panel = edit_panel

        self.edit_session: EditSession | None = None
        self._pending_adjustments: tuple[float, float, float] | None = None
        self._adjustment_timer = QTimer(parent_widget)
        self._adjustment_timer.setSingleShot(True)
        self._adjustment_timer.setInterval(30)
        self._adjustment_timer.timeout.connect(self._apply_pending_adjustments)

        self._connect_edit_panel()

    def _connect_edit_panel(self) -> None:
        panel = self.edit_panel

        panel.rotate_cw.connect(lambda: self._apply_edit(lambda es: es.rotate(clockwise=True)))
        panel.rotate_ccw.connect(lambda: self._apply_edit(lambda es: es.rotate(clockwise=False)))
        panel.flip_horizontal.connect(lambda: self._apply_edit(lambda es: es.flip_horizontal()))
        panel.flip_vertical.connect(lambda: self._apply_edit(lambda es: es.flip_vertical()))

        # crop_mode_toggled is deliberately NOT connected here: MainWindow
        # connects it to a small wrapper that also exits face-edit mode
        # first (the two modes can't both be active, since both interpret
        # mouse drags on the shared ImageViewer), then calls set_crop_mode
        # below itself.

        panel.adjustments_changed.connect(self._on_adjustments_changed)
        panel.adjustments_committed.connect(self._on_adjustments_committed)

        panel.resize_requested.connect(
            lambda w, h: self._apply_edit(lambda es: es.resize((w, h)))
        )

        panel.undo_requested.connect(lambda: self._apply_edit(lambda es: es.undo()))
        panel.redo_requested.connect(lambda: self._apply_edit(lambda es: es.redo()))
        panel.reset_requested.connect(self._on_reset_edits)

        # save_*_requested are deliberately NOT connected here either: an
        # overwrite save invalidates that photo's cached face data (the old
        # boxes/embeddings no longer match the new pixel geometry), so
        # MainWindow connects all three save signals to one wrapper that
        # calls save() below and then handles that side effect for the
        # overwrite case, matching this class's existing pattern for
        # cross-feature concerns.

    # -- Session lifecycle, called by MainWindow at the right moments --------

    def has_unsaved_edits(self) -> bool:
        return self.edit_session is not None and self.edit_session.has_edits()

    def unsaved_edits_photo_name(self) -> str:
        assert self.edit_session is not None and self.edit_session.source_path is not None
        return self.edit_session.source_path.name

    def discard(self) -> None:
        self.edit_session = None

    def on_new_photo_loaded(self, qimage: QImage) -> None:
        """Called once a newly-navigated-to photo's full-resolution image has
        finished loading -- resets edit-panel widgets and crop mode, which
        shouldn't silently carry over from whatever photo was showing before."""
        self.edit_panel.set_image_size(qimage.width(), qimage.height())
        self.edit_panel.reset_adjustment_sliders()
        self.edit_panel.set_history_enabled(False, False)
        self.viewer.set_crop_mode(False)
        self.edit_panel.set_crop_mode_active(False)

    def on_tab_activated(self) -> None:
        if self.ensure_session() is not None:
            self.refresh_preview()

    def on_tab_deactivated(self) -> None:
        self.exit_crop_mode()

    def exit_crop_mode(self) -> None:
        self.viewer.set_crop_mode(False)
        self.edit_panel.set_crop_mode_active(False)

    # -- Core session operations ----------------------------------------

    def ensure_session(self) -> EditSession | None:
        item = self.library.current_item
        if item is None:
            return None
        if self.edit_session is None or self.edit_session.source_path != item.path:
            self.edit_session = EditSession.from_path(item.path)
            self.edit_panel.reset_adjustment_sliders()
        return self.edit_session

    def refresh_preview(self) -> None:
        if self.edit_session is None:
            return
        image = self.edit_session.render()
        self.viewer.set_image(pil_to_qimage(image))
        self.edit_panel.set_image_size(image.width, image.height)
        self.edit_panel.set_history_enabled(self.edit_session.can_undo(), self.edit_session.can_redo())

    def _apply_edit(self, fn) -> None:
        session = self.ensure_session()
        if session is None:
            return
        fn(session)
        self.refresh_preview()

    def set_crop_mode(self, enabled: bool) -> None:
        if enabled:
            self.ensure_session()
        self.viewer.set_crop_mode(enabled)

    def on_crop_selected(self, box: tuple[int, int, int, int]) -> None:
        session = self.ensure_session()
        if session is None:
            return
        session.crop(box)
        self.viewer.set_crop_mode(False)
        self.edit_panel.set_crop_mode_active(False)
        self.refresh_preview()

    def _on_adjustments_changed(self, brightness: float, contrast: float, saturation: float) -> None:
        if self.ensure_session() is None:
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
        self.refresh_preview()

    def _on_adjustments_committed(self) -> None:
        if self.edit_session is None:
            return
        if self._adjustment_timer.isActive():
            self._adjustment_timer.stop()
            self._apply_pending_adjustments()
        self.edit_session.commit_adjustments()
        self.refresh_preview()

    def _on_reset_edits(self) -> None:
        if self.edit_session is None:
            return
        self.edit_session.reset()
        self.edit_panel.reset_adjustment_sliders()
        self.refresh_preview()
        self.edit_panel.set_history_enabled(False, False)

    def save(self, mode: str) -> Path | None:
        """Returns the saved path on success, or None if the user cancelled
        or the save failed (already reported to them). Callers that need to
        react to a successful *overwrite* specifically (invalidating cached
        face data, since the pre-edit pixel geometry it was computed against
        no longer matches) should compare mode == "overwrite" and check the
        return value.
        """
        session = self.edit_session
        if session is None or not session.has_edits():
            self.edit_panel.set_status_message("No edits to save.")
            return None

        try:
            if mode == "copy":
                saved_path = session.save(overwrite=False)
            elif mode == "overwrite":
                confirm = QMessageBox.question(
                    self.parent_widget,
                    "Overwrite Original",
                    f"Overwrite {session.source_path.name} with the edited version? "
                    "This cannot be undone.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return None
                saved_path = session.save(overwrite=True)
            elif mode == "as":
                default_name = str(
                    session.source_path.with_name(
                        f"{session.source_path.stem}_edited{session.source_path.suffix}"
                    )
                )
                chosen, _ = QFileDialog.getSaveFileName(self.parent_widget, "Save As", default_name)
                if not chosen:
                    return None
                saved_path = session.save(path=Path(chosen))
            else:
                return None
        except OSError as exc:
            QMessageBox.critical(self.parent_widget, "Save Failed", str(exc))
            return None

        self.edit_panel.set_status_message(f"Saved to {saved_path}")
        return saved_path
