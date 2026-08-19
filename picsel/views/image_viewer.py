"""Zoomable/pannable image display with an optional crop-selection mode."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)


class _RubberBandDrawer:
    """Shared "drag out a rectangle" state machine behind crop-selection and
    manual face-box drawing on ImageViewer -- both interpret mouse
    press/move/release on the same scene identically, differing only in pen
    color and whether the drawn rect item is cleared immediately after a
    valid release. Crop's item deliberately stays visible after release
    (ImageViewer's caller turns crop mode off once it's handled
    crop_selected, which is what actually clears it via set_crop_mode);
    a drawn face box has no such caller-driven lifecycle, so it always
    clears itself right away.
    """

    def __init__(self, scene: QGraphicsScene, pixmap_item: QGraphicsPixmapItem, pen_color: str, clear_after_release: bool) -> None:
        self._scene = scene
        self._pixmap_item = pixmap_item
        self._pen_color = pen_color
        self._clear_after_release = clear_after_release
        self.item: QGraphicsRectItem | None = None
        self._origin: QRectF | None = None

    def clear(self) -> None:
        if self.item is not None:
            self._scene.removeItem(self.item)
            self.item = None
        self._origin = None

    def start(self, scene_pos) -> None:
        self.clear()
        self._origin = scene_pos
        self.item = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
        pen = QPen(QColor(self._pen_color))
        pen.setWidth(2)
        pen.setCosmetic(True)
        self.item.setPen(pen)
        self._scene.addItem(self.item)

    def update(self, scene_pos) -> None:
        rect = QRectF(self._origin, scene_pos).normalized()
        rect = rect.intersected(self._pixmap_item.boundingRect())
        self.item.setRect(rect)

    def finish(self) -> tuple[int, int, int, int] | None:
        """Returns the drawn box (left, top, right, bottom) if it's large
        enough to be a real selection, else None -- callers should only emit
        their signal in the non-None case, matching the pre-refactor
        behavior of silently discarding an effectively-zero-size drag."""
        rect = self.item.rect()
        box = (int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))
        self._origin = None
        valid = rect.width() >= 1 and rect.height() >= 1
        if self._clear_after_release:
            self.clear()
        return box if valid else None


class ImageViewer(QGraphicsView):
    """Displays an image with mouse-wheel zoom, drag-to-pan, and crop selection."""

    crop_selected = Signal(tuple)  # (left, top, right, bottom) in image pixel coords
    face_box_added = Signal(tuple)  # (left, top, right, bottom) drawn by the user in face-edit mode
    face_box_dismiss_requested = Signal(int)  # index into the list last passed to set_face_boxes

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # Set once the user has manually zoomed (wheel), so resizeEvent stops
        # force-refitting the image on every viewport resize -- that resize
        # isn't only the window resizing, it's also the splitter between the
        # image and the side panel (or the filmstrip) being nudged, which
        # otherwise silently threw away a deliberate zoom-in (e.g. to check
        # focus) any time the user touched an unrelated splitter. Reset on
        # each new image, so browsing to the next photo still starts fit to
        # the window like before; double-click resets it early too.
        self._user_zoomed = False

        self._crop_mode = False
        self._crop_drawer = _RubberBandDrawer(self._scene, self._pixmap_item, "yellow", clear_after_release=False)

        self._face_edit_mode = False
        self._face_box_items: list[QGraphicsRectItem] = []
        self._face_drawer = _RubberBandDrawer(self._scene, self._pixmap_item, "cyan", clear_after_release=True)

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#202020"))

    def set_image(self, image: QImage | QPixmap) -> None:
        pixmap = QPixmap.fromImage(image) if isinstance(image, QImage) else image
        self._crop_drawer.clear()
        self._face_drawer.clear()
        self.set_face_boxes([])  # a new image's face boxes haven't been (re)computed yet
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._user_zoomed = False
        self.fit_to_window()

    def fit_to_window(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit_to_window()

    def wheelEvent(self, event) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._user_zoomed = True
        self.scale(zoom_factor, zoom_factor)

    def mouseDoubleClickEvent(self, event) -> None:
        # The standard "reset zoom" convention, and the only deliberate way
        # to get back to fit-to-window now that resizeEvent no longer does
        # it as an (accidental) side effect once the user has zoomed.
        if event.button() == Qt.MouseButton.LeftButton and not self._crop_mode and not self._face_edit_mode:
            self._user_zoomed = False
            self.fit_to_window()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # -- Face box overlay -------------------------------------------------

    def set_face_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        """Draw `boxes` (left, top, right, bottom, in this image's own pixel
        coordinates -- same space `set_image` was called with) as an overlay,
        replacing any previously drawn boxes. In face-edit mode, clicking
        inside one of these boxes emits `face_box_dismiss_requested` with its
        index in `boxes`.
        """
        for item in self._face_box_items:
            self._scene.removeItem(item)
        self._face_box_items = []

        pen = QPen(QColor("yellow"))
        pen.setWidth(2)
        pen.setCosmetic(True)
        # A visible (if faint) fill, not just an outline: QGraphicsRectItem's
        # hit-testing only covers the stroke unless there's a brush, so an
        # unfilled box would only respond to clicks on its exact border.
        brush = QColor(255, 220, 0, 40)
        for left, top, right, bottom in boxes:
            rect_item = QGraphicsRectItem(QRectF(left, top, right - left, bottom - top))
            rect_item.setPen(pen)
            rect_item.setBrush(brush)
            self._scene.addItem(rect_item)
            self._face_box_items.append(rect_item)

    def set_face_edit_mode(self, enabled: bool) -> None:
        self._face_edit_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._face_drawer.clear()

    # -- Crop selection -------------------------------------------------

    def set_crop_mode(self, enabled: bool) -> None:
        self._crop_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._crop_drawer.clear()

    def mousePressEvent(self, event) -> None:
        if self._face_edit_mode and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            clicked = self.scene().itemAt(scene_pos, self.transform())
            if clicked in self._face_box_items:
                self.face_box_dismiss_requested.emit(self._face_box_items.index(clicked))
                event.accept()
                return
            self._face_drawer.start(scene_pos)
            event.accept()
            return
        if self._crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self._crop_drawer.start(self.mapToScene(event.pos()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._face_edit_mode and self._face_drawer.item is not None:
            self._face_drawer.update(self.mapToScene(event.pos()))
            event.accept()
            return
        if self._crop_mode and self._crop_drawer.item is not None:
            self._crop_drawer.update(self.mapToScene(event.pos()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._face_edit_mode and self._face_drawer.item is not None:
            box = self._face_drawer.finish()
            if box is not None:
                self.face_box_added.emit(box)
            event.accept()
            return
        if self._crop_mode and self._crop_drawer.item is not None:
            box = self._crop_drawer.finish()
            if box is not None:
                self.crop_selected.emit(box)
            event.accept()
            return
        super().mouseReleaseEvent(event)
