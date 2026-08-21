from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap

import tamis.views.thumbnail_list as thumbnail_list_module
from tamis.models.image_item import ImageItem, Status
from tamis.quality.store import PhotoScores
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
    # Item text is the filename alone: the star rating and the aesthetic
    # score are painted by _ThumbnailDelegate on their own lines, so keeping
    # them out of the text is what stops them being drawn twice.
    assert thumbnail_list.item(2).text() == "img2.jpg"
    assert thumbnail_list.item(2).data(Qt.ItemDataRole.UserRole).rating == 4
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
    assert all(thumbnail_list.item(i).text() == f"img{i}.jpg" for i in range(4))
    assert all(thumbnail_list.item(i).data(Qt.ItemDataRole.UserRole).rating == 5 for i in range(4))


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


def test_scores_are_stored_and_shown_per_item(qapp):
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path(f"/tmp/img{i}.jpg")) for i in range(3)]
    thumbnail_list.set_items(items)

    thumbnail_list.set_scores({items[0].path: PhotoScores(quality=80, blur=90), items[2].path: PhotoScores(quality=20, blur=90)})

    assert thumbnail_list.score_for(items[0]).quality == 80
    assert thumbnail_list.score_for(items[1]) is None
    assert thumbnail_list.item(0).data(thumbnail_list_module._SCORE_ROLE).quality == 80


def test_the_filter_hides_low_scoring_photos_without_removing_them(qapp):
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path(f"/tmp/img{i}.jpg")) for i in range(3)]
    thumbnail_list.set_items(items)
    thumbnail_list.set_scores({items[0].path: PhotoScores(quality=80, blur=90), items[1].path: PhotoScores(quality=30, blur=90), items[2].path: PhotoScores(quality=55, blur=90)})

    thumbnail_list.set_min_score(50)

    assert thumbnail_list.count() == 3  # hidden, not removed
    assert not thumbnail_list.item(0).isHidden()
    assert thumbnail_list.item(1).isHidden()
    assert not thumbnail_list.item(2).isHidden()

    thumbnail_list.set_min_score(0)
    assert not any(thumbnail_list.item(i).isHidden() for i in range(3))


def test_an_unscored_photo_is_never_hidden(qapp):
    # Scoring runs in the background; photos must not vanish as results
    # trickle in, only once they are known to be below the cutoff.
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path("/tmp/img0.jpg"))]
    thumbnail_list.set_items(items)
    thumbnail_list.set_min_score(90)
    assert not thumbnail_list.is_filtered_out(items[0])


def test_scores_survive_a_re_sort(qapp):
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path(f"/tmp/img{i}.jpg")) for i in range(3)]
    thumbnail_list.set_items(items)
    thumbnail_list.set_scores({items[i].path: PhotoScores(quality=10 * i, blur=90) for i in range(3)})

    thumbnail_list.set_items(list(reversed(items)))

    assert thumbnail_list.item(0).data(thumbnail_list_module._SCORE_ROLE).quality == 20
    assert thumbnail_list.score_for(items[1]).quality == 10


def test_the_delegate_reserves_room_for_three_text_lines(qapp):
    # Without this the item is sized for a single line and the filename gets
    # painted on top of the thumbnail.
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QStyleOptionViewItem

    thumbnail_list = ThumbnailList()
    thumbnail_list.set_items([ImageItem(path=Path("/tmp/img0.jpg"))])
    option = QStyleOptionViewItem()
    option.font = thumbnail_list.font()
    hint = thumbnail_list.itemDelegate().sizeHint(option, thumbnail_list.model().index(0, 0))

    line = QFontMetrics(thumbnail_list.font()).height()
    assert hint.height() >= ICON_SIZE.height() + 3 * line
    assert hint == thumbnail_list.gridSize()  # delegate and grid must agree


def test_rebuilding_the_list_does_not_look_like_the_user_picking_a_photo(qapp):
    """clear() does not simply drop to "no current row".

    As rows are removed Qt walks the current row along, emitting
    currentRowChanged with intermediate *valid* indices. The owning window
    reads that signal as a photo being selected, so an unblocked rebuild
    silently navigates the library -- which showed up as the displayed photo
    changing on its own while the filmstrip was re-sorted underneath it.
    """
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path(f"/tmp/img{i}.jpg")) for i in range(12)]
    thumbnail_list.set_items(items)
    thumbnail_list.setCurrentRow(7)

    seen = []
    thumbnail_list.currentRowChanged.connect(seen.append)
    thumbnail_list.set_items(list(reversed(items)))

    assert seen == [], f"rebuild emitted currentRowChanged{seen}"


def test_signal_blocking_is_restored_after_a_rebuild(qapp):
    # Blocking must not leak: a later genuine selection has to be heard.
    thumbnail_list = ThumbnailList()
    items = [ImageItem(path=Path(f"/tmp/img{i}.jpg")) for i in range(4)]
    thumbnail_list.set_items(items)
    assert not thumbnail_list.signalsBlocked()

    seen = []
    thumbnail_list.currentRowChanged.connect(seen.append)
    thumbnail_list.setCurrentRow(2)
    assert seen == [2]


def test_the_grid_and_the_delegate_derive_the_cell_size_from_one_place(qapp):
    # They used to compute it separately from the same formula, which is a
    # standing invitation to change one and not the other -- and a mismatch
    # paints the filename on top of the thumbnail rather than failing loudly.
    from PySide6.QtWidgets import QStyleOptionViewItem

    thumbnail_list = ThumbnailList()
    thumbnail_list.set_items([ImageItem(path=Path("/tmp/img0.jpg"))])
    option = QStyleOptionViewItem()
    option.font = thumbnail_list.font()

    hint = thumbnail_list.itemDelegate().sizeHint(option, thumbnail_list.model().index(0, 0))
    assert hint == thumbnail_list.gridSize()
    assert hint == thumbnail_list_module._cell_size(thumbnail_list.font())


def test_the_cell_is_no_taller_than_its_contents_need(qapp):
    # The filmstrip competes with the image for window height, so the padding
    # is deliberately tight.
    from PySide6.QtGui import QFontMetrics

    thumbnail_list = ThumbnailList()
    line = QFontMetrics(thumbnail_list.font()).height()
    content = ICON_SIZE.height() + 3 * line
    assert thumbnail_list.gridSize().height() - content <= 6
