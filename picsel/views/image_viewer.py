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


class ImageViewer(QGraphicsView):
    """Displays an image with mouse-wheel zoom, drag-to-pan, and crop selection."""

    crop_selected = Signal(tuple)  # (left, top, right, bottom) in image pixel coords

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._crop_mode = False
        self._crop_rect_item: QGraphicsRectItem | None = None
        self._crop_origin = None

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#202020"))

    def set_image(self, image: QImage | QPixmap) -> None:
        pixmap = QPixmap.fromImage(image) if isinstance(image, QImage) else image
        self._clear_crop_rect()
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_window()

    def fit_to_window(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_to_window()

    def wheelEvent(self, event) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)

    # -- Crop selection -------------------------------------------------

    def set_crop_mode(self, enabled: bool) -> None:
        self._crop_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._clear_crop_rect()

    def _clear_crop_rect(self) -> None:
        if self._crop_rect_item is not None:
            self._scene.removeItem(self._crop_rect_item)
            self._crop_rect_item = None

    def mousePressEvent(self, event) -> None:
        if self._crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self._crop_origin = self.mapToScene(event.pos())
            self._clear_crop_rect()
            self._crop_rect_item = QGraphicsRectItem(QRectF(self._crop_origin, self._crop_origin))
            pen = QPen(QColor("yellow"))
            pen.setWidth(2)
            pen.setCosmetic(True)
            self._crop_rect_item.setPen(pen)
            self._scene.addItem(self._crop_rect_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._crop_mode and self._crop_rect_item is not None and self._crop_origin is not None:
            current = self.mapToScene(event.pos())
            rect = QRectF(self._crop_origin, current).normalized()
            rect = rect.intersected(self._pixmap_item.boundingRect())
            self._crop_rect_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._crop_mode and self._crop_rect_item is not None:
            rect = self._crop_rect_item.rect()
            box = (int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))
            self._crop_origin = None
            if rect.width() >= 1 and rect.height() >= 1:
                self.crop_selected.emit(box)
            event.accept()
            return
        super().mouseReleaseEvent(event)
