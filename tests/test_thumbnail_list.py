from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap

import tamis.views.thumbnail_list as thumbnail_list_module
from tamis.models.image_item import ImageItem, Status
from tamis.views.thumbnail_list import ICON_SIZE, _RAW_PIXMAP_ROLE, ThumbnailList, _badged_pixmap


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


def test_refresh_item_redraws_only_the_row_asked_for(qapp):
    # Marking or rating changes exactly one photo. Redrawing the whole
    # filmstrip made the cost of the app's most repeated keystrokes scale with
    # the folder size and with how much of it was already marked.
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path(f"/tmp/img{i}.jpg")) for i in range(5)]
    thumbnail_list.set_items(items)
    pixmap = QPixmap(ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.gray)
    for i in range(thumbnail_list.count()):
        thumbnail_list.item(i).setData(_RAW_PIXMAP_ROLE, pixmap)

    repainted = []
    original = thumbnail_list_module._badged_pixmap
    thumbnail_list_module._badged_pixmap = lambda pm, status: repainted.append(status) or original(pm, status)
    try:
        items[2].status = Status.SELECTED
        items[2].rating = 4
        thumbnail_list.refresh_item(2)
    finally:
        thumbnail_list_module._badged_pixmap = original

    assert len(repainted) == 1
    assert thumbnail_list.item(2).text() == "img2.jpg\n★★★★"
    assert thumbnail_list.item(0).text() == "img0.jpg"


def test_refresh_item_ignores_an_out_of_range_index(qapp):
    thumbnail_list = ThumbnailList()
    thumbnail_list.set_items([ImageItem(path=Path("/tmp/a.jpg"))])
    thumbnail_list.refresh_item(99)  # must not raise
    thumbnail_list.refresh_item(-1)


def test_refresh_badges_still_redraws_every_row(qapp):
    # set_items and re-sorts legitimately need the whole list rebuilt.
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path(f"/tmp/img{i}.jpg"), rating=i) for i in range(4)]
    thumbnail_list.set_items(items)
    for item in items:
        item.rating = 5
    thumbnail_list.refresh_badges()
    assert all(thumbnail_list.item(i).text().endswith("★★★★★") for i in range(4))


def test_set_items_reuses_already_decoded_thumbnails(qapp):
    # Re-sorting calls set_items with the same photos in a new order; without
    # a cache that re-decoded the entire folder from disk.
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path(f"/tmp/img{i}.jpg")) for i in range(4)]
    thumbnail_list.set_items(items)
    started = len(thumbnail_list._pending_workers)
    assert started == 4

    for item in items:
        thumbnail_list._pixmap_cache[item.path] = _solid_pixmap()
    thumbnail_list._pending_workers.clear()
    thumbnail_list.set_items(list(reversed(items)))

    assert thumbnail_list._pending_workers == []
    assert all(thumbnail_list.item(i).icon() is not None for i in range(4))


def test_set_items_drops_cached_thumbnails_for_photos_no_longer_present(qapp):
    # Otherwise the cache grows across every folder opened in a session.
    thumbnail_list = ThumbnailList()
    old_items = [ImageItem(path=Path(f"/tmp/old{i}.jpg")) for i in range(3)]
    thumbnail_list.set_items(old_items)
    for item in old_items:
        thumbnail_list._pixmap_cache[item.path] = _solid_pixmap()

    thumbnail_list.set_items([ImageItem(path=Path("/tmp/new0.jpg"))])

    assert thumbnail_list._pixmap_cache == {}


def test_reload_item_forgets_the_cached_thumbnail(qapp):
    # An overwrite save changes the file's pixels under a cached thumbnail.
    thumbnail_list = ThumbnailList()
    item = ImageItem(path=Path("/tmp/img0.jpg"))
    thumbnail_list.set_items([item])
    thumbnail_list._pixmap_cache[item.path] = _solid_pixmap()
    thumbnail_list._pending_workers.clear()

    thumbnail_list.reload_item(0)

    assert item.path not in thumbnail_list._pixmap_cache
    assert len(thumbnail_list._pending_workers) == 1


def test_reload_item_ignores_an_out_of_range_index(qapp):
    thumbnail_list = ThumbnailList()
    thumbnail_list.set_items([ImageItem(path=Path("/tmp/a.jpg"))])
    thumbnail_list.reload_item(42)  # must not raise
