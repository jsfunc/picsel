from pathlib import Path

from PySide6.QtGui import QColor, QPixmap

from tamis.models.image_item import ImageItem, Status
from tamis.views.thumbnail_list import _RAW_PIXMAP_ROLE, ThumbnailList, _badged_pixmap


def _solid_pixmap(color=(80, 80, 80), size=60) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*color))
    return pixmap


def test_badged_pixmap_leaves_unrated_thumbnails_unchanged(qapp):
    pixmap = _solid_pixmap()
    badged = _badged_pixmap(pixmap, Status.UNRATED)
    assert badged is pixmap  # no badge drawn -- returns the same object, not just visually identical


def test_badged_pixmap_draws_a_visibly_different_corner_for_selected(qapp):
    pixmap = _solid_pixmap(color=(80, 80, 80))
    badged = _badged_pixmap(pixmap, Status.SELECTED)

    original_corner = pixmap.toImage().pixelColor(14, 14)
    badged_corner = badged.toImage().pixelColor(14, 14)
    assert badged_corner != original_corner  # a check-mark badge was actually drawn there


def test_badged_pixmap_draws_a_visibly_different_corner_for_rejected(qapp):
    pixmap = _solid_pixmap(color=(80, 80, 80))
    badged = _badged_pixmap(pixmap, Status.REJECTED)

    original_corner = pixmap.toImage().pixelColor(14, 14)
    badged_corner = badged.toImage().pixelColor(14, 14)
    assert badged_corner != original_corner


def test_badged_pixmap_selected_and_rejected_are_visibly_different_from_each_other(qapp):
    # The whole point is a cue that doesn't rely on color -- but as long as
    # color *is* used too, selected and rejected shouldn't render identically.
    pixmap = _solid_pixmap(color=(80, 80, 80))
    selected = _badged_pixmap(pixmap, Status.SELECTED)
    rejected = _badged_pixmap(pixmap, Status.REJECTED)
    assert selected.toImage().pixelColor(14, 14) != rejected.toImage().pixelColor(14, 14)


def test_refresh_badges_redraws_the_icon_when_status_changes(qapp):
    thumbnail_list = ThumbnailList()
    item = ImageItem(path=Path("/some/folder/photo.jpg"))
    thumbnail_list.set_items([item])

    list_item = thumbnail_list.item(0)
    raw = _solid_pixmap(color=(80, 80, 80))
    list_item.setData(_RAW_PIXMAP_ROLE, raw)
    list_item.setIcon(_badged_pixmap(raw, item.status))  # simulates what _on_thumbnail_ready does

    unrated_icon_pixmap = list_item.icon().pixmap(60, 60)

    item.status = Status.SELECTED
    thumbnail_list.refresh_badges()

    selected_icon_pixmap = list_item.icon().pixmap(60, 60)
    assert selected_icon_pixmap.toImage().pixelColor(14, 14) != unrated_icon_pixmap.toImage().pixelColor(14, 14)
