"""Horizontal filmstrip of thumbnails with async loading and status badges."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QThreadPool
from PySide6.QtGui import QFontMetrics, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem

from tamis.models.image_item import ImageItem, Status
from tamis.thumbnails import ThumbnailWorker
from tamis.views.theme import (
    BADGE_TEXT_COLOR,
    NEUTRAL_TINT,
    REJECTED_BADGE_COLOR,
    REJECTED_TINT,
    SELECTED_BADGE_COLOR,
    SELECTED_TINT,
)

ICON_SIZE = QSize(120, 120)

# The un-badged thumbnail pixmap, cached per item so a status change can
# redraw the badge without re-decoding the image from disk.
_RAW_PIXMAP_ROLE = Qt.ItemDataRole.UserRole + 1

BADGE_DIAMETER = 22


def _badged_pixmap(pixmap: QPixmap, status: Status) -> QPixmap:
    """A copy of `pixmap` with a small check/cross badge in the top-left
    corner for selected/rejected status -- the background tint alone isn't
    reliably distinguishable for colorblind users (red/green is the most
    common form of color vision deficiency), so this adds a shape cue that
    doesn't depend on color to read."""
    if status is Status.UNRATED:
        return pixmap
    color, glyph = (SELECTED_BADGE_COLOR, "✓") if status is Status.SELECTED else (REJECTED_BADGE_COLOR, "✕")

    badged = QPixmap(pixmap)
    painter = QPainter(badged)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    circle = QRectF(3, 3, BADGE_DIAMETER, BADGE_DIAMETER)
    painter.setPen(BADGE_TEXT_COLOR)
    painter.setBrush(color)
    painter.drawEllipse(circle)
    font = painter.font()
    font.setPixelSize(int(BADGE_DIAMETER * 0.65))
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(circle, Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return badged


class ThumbnailList(QListWidget):
    """Single-row filmstrip; `currentRowChanged` reflects the selected image's index."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setIconSize(ICON_SIZE)
        # Thumbnails load asynchronously, so an item's size hint changes once its icon
        # arrives (no icon -> icon). setUniformItemSizes(True) would cache the smaller
        # pre-icon layout rect and never grow it, squashing the thumbnail into a sliver.
        self.setUniformItemSizes(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Item text can span two lines (filename + star rating); size the grid and
        # the widget's height around that so items aren't clipped at the bottom.
        line_height = QFontMetrics(self.font()).height()
        item_size = QSize(ICON_SIZE.width() + 20, ICON_SIZE.height() + 2 * line_height + 12)
        self.setGridSize(item_size)
        scrollbar_height = self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(item_size.height() + scrollbar_height + 2 * self.frameWidth() + 4)

        self._thread_pool = QThreadPool.globalInstance()
        self._pending_workers: list[ThumbnailWorker] = []
        self._generation = 0
        # Decoded thumbnails, kept across set_items so a re-sort doesn't
        # re-decode the folder. Pruned to the current photos on every
        # set_items, so it stays bounded by folder size (~57KB each).
        self._pixmap_cache: dict[Path, QPixmap] = {}

    def set_items(self, items: list[ImageItem]) -> None:
        self._generation += 1
        generation = self._generation
        # Re-sorting calls this with the same photos in a different order, and
        # rebuilding the list drops every decoded thumbnail -- so every sort
        # used to re-decode the whole folder from disk (measured: 565 decodes,
        # 1941ms, to reproduce work already done). Keep what we already have,
        # pruned to the photos actually present, which also releases the old
        # folder's thumbnails on a folder switch rather than growing forever.
        incoming = {item.path for item in items}
        self._pixmap_cache = {
            path: pixmap for path, pixmap in self._pixmap_cache.items() if path in incoming
        }
        # Don't clear _pending_workers here: workers from the previous folder may
        # still be running on the thread pool. Dropping their only Python reference
        # would let their `signals` QObject get collected mid-flight, crashing the
        # worker thread when it later tries to emit. Stale results are already
        # discarded by the generation check in _on_thumbnail_ready, which also
        # removes each worker from this list once it actually completes.
        self.clear()
        for item in items:
            list_item = QListWidgetItem(item.name)
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            list_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.addItem(list_item)
            self._request_thumbnail(list_item, item, generation)
        self.refresh_badges()

    def _request_thumbnail(self, list_item: QListWidgetItem, item: ImageItem, generation: int) -> None:
        cached = self._pixmap_cache.get(item.path)
        if cached is not None:
            # Only the raw pixmap: set_items ends with refresh_badges(), which
            # draws the badge for every row. Compositing it here as well would
            # do that twice for the whole folder on every re-sort.
            list_item.setData(_RAW_PIXMAP_ROLE, cached)
            return
        worker = ThumbnailWorker(item.path)
        self._pending_workers.append(worker)
        worker.signals.finished.connect(
            lambda path, image, error, li=list_item, w=worker, gen=generation: self._on_thumbnail_ready(
                li, image, error, w, gen
            )
        )
        self._thread_pool.start(worker)

    def _on_thumbnail_ready(
        self, list_item: QListWidgetItem, image: QImage, error: str, worker: ThumbnailWorker, generation: int
    ) -> None:
        if worker in self._pending_workers:
            self._pending_workers.remove(worker)
        if generation != self._generation:
            return
        if image.isNull():
            list_item.setToolTip(f"Failed to load thumbnail: {error}" if error else "Failed to load thumbnail")
            return
        pixmap = QPixmap.fromImage(image)
        img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
        self._pixmap_cache[img_item.path] = pixmap
        self._apply_pixmap(list_item, pixmap)

    def _apply_pixmap(self, list_item: QListWidgetItem, pixmap: QPixmap) -> None:
        list_item.setData(_RAW_PIXMAP_ROLE, pixmap)
        img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
        list_item.setIcon(QIcon(_badged_pixmap(pixmap, img_item.status)))

    def reload_item(self, index: int) -> None:
        """Re-decode one row's thumbnail because its file changed on disk.

        Needed after an overwrite save: the displayed thumbnail was decoded
        from the pre-edit pixels. That was already stale before thumbnails
        were cached (nothing re-read the file until the list was rebuilt);
        caching would have made it stale until a folder switch, so the
        invalidation is wired up rather than left implicit.
        """
        list_item = self.item(index)
        if list_item is None:
            return
        img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
        self._pixmap_cache.pop(img_item.path, None)
        self._request_thumbnail(list_item, img_item, self._generation)

    def refresh_item(self, index: int) -> None:
        """Redraw one row's label, tint and badge.

        Marking and rating change exactly one photo, and used to go through
        `refresh_badges()`, which recomposites a badged pixmap for every
        thumbnail in the folder. That made the cost of the app's most repeated
        keystrokes scale with the size of the folder *and* with how much of it
        had already been marked -- `_badged_pixmap` returns immediately for an
        unrated photo, so the work grew as culling progressed. Measured on a
        584-photo folder with every thumbnail decoded: 5.9ms per keypress at
        the start of a pass, 26.4ms once everything was marked, for a redraw
        of one row.
        """
        list_item = self.item(index)
        if list_item is None:
            return
        img_item: ImageItem = list_item.data(Qt.ItemDataRole.UserRole)
        label = img_item.name
        if img_item.rating:
            label += "\n" + "★" * img_item.rating
        list_item.setText(label)
        if img_item.status is Status.SELECTED:
            list_item.setBackground(SELECTED_TINT)
        elif img_item.status is Status.REJECTED:
            list_item.setBackground(REJECTED_TINT)
        else:
            list_item.setBackground(NEUTRAL_TINT)
        raw_pixmap = list_item.data(_RAW_PIXMAP_ROLE)
        if raw_pixmap is not None:
            list_item.setIcon(QIcon(_badged_pixmap(raw_pixmap, img_item.status)))

    def refresh_badges(self) -> None:
        """Redraw every row. Only needed when the whole list changed (a new
        folder, or a re-sort); a single photo's change should use
        `refresh_item`."""
        for i in range(self.count()):
            self.refresh_item(i)

    def select_index(self, index: int) -> None:
        if 0 <= index < self.count() and self.currentRow() != index:
            self.blockSignals(True)
            self.setCurrentRow(index)
            self.blockSignals(False)
        if 0 <= index < self.count():
            self.scrollToItem(self.item(index), QAbstractItemView.ScrollHint.EnsureVisible)
