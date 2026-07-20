import json
from pathlib import Path

import pytest
from PIL import Image

from picsel.models import ImageItem, ImageLibrary, Status


def _make_image(path: Path, size=(4, 4), color=(255, 0, 0)) -> None:
    Image.new("RGB", size, color).save(path)


def test_image_item_defaults():
    item = ImageItem(path=Path("/some/folder/photo.jpg"))
    assert item.name == "photo.jpg"
    assert item.status is Status.UNRATED
    assert item.rating == 0


def test_library_load_filters_supported_extensions(tmp_path):
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "b.png")
    (tmp_path / "notes.txt").write_text("not an image")

    library = ImageLibrary()
    library.load(tmp_path)

    names = sorted(item.name for item in library.items)
    assert names == ["a.jpg", "b.png"]
    assert library.current_index == 0
    assert library.current_item is not None


def test_library_navigation_is_clamped(tmp_path):
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "b.jpg")
    library = ImageLibrary()
    library.load(tmp_path)

    library.prev()
    assert library.current_index == 0

    library.next()
    assert library.current_index == 1
    library.next()
    assert library.current_index == 1


def test_library_state_roundtrip(tmp_path):
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "b.jpg")

    library = ImageLibrary()
    library.load(tmp_path)
    library.set_status(0, Status.SELECTED)
    library.set_rating(0, 4)
    library.set_status(1, Status.REJECTED)
    library.save_state()

    reloaded = ImageLibrary()
    reloaded.load(tmp_path)
    by_name = {item.name: item for item in reloaded.items}
    assert by_name["a.jpg"].status is Status.SELECTED
    assert by_name["a.jpg"].rating == 4
    assert by_name["b.jpg"].status is Status.REJECTED

    state = json.loads((tmp_path / ".picsel_state.json").read_text())
    assert state["a.jpg"] == {"status": "selected", "rating": 4}


def test_library_counts(tmp_path):
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "b.jpg")
    _make_image(tmp_path / "c.jpg")
    library = ImageLibrary()
    library.load(tmp_path)
    library.set_status(0, Status.SELECTED)
    library.set_status(1, Status.REJECTED)

    assert library.counts() == {"selected": 1, "rejected": 1, "unrated": 1}


def test_register_name_use_increments_per_name(tmp_path):
    library = ImageLibrary()
    library.load(tmp_path)
    assert library.register_name_use("trip") == 1
    assert library.register_name_use("trip") == 2
    assert library.register_name_use("other") == 1


def test_load_raises_oserror_and_leaves_previous_state_untouched(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    _make_image(good / "a.jpg")

    library = ImageLibrary()
    library.load(good)
    assert library.folder == good
    assert len(library.items) == 1

    missing = tmp_path / "does_not_exist"
    with pytest.raises(OSError):
        library.load(missing)

    # A failed load must not clobber the previously loaded folder/items —
    # otherwise a later save_state() would write into the folder that failed.
    assert library.folder == good
    assert len(library.items) == 1


def test_sort_items_reorders_and_preserves_current_selection(tmp_path):
    _make_image(tmp_path / "b.jpg")
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "c.jpg")

    library = ImageLibrary()
    library.load(tmp_path)  # loaded name-sorted: a, b, c
    current = next(item for item in library.items if item.name == "b.jpg")
    library.current_index = library.items.index(current)
    assert [item.name for item in library.items] == ["a.jpg", "b.jpg", "c.jpg"]

    library.sort_items(key=lambda item: -ord(item.name[0]))  # descending -> c, b, a

    assert [item.name for item in library.items] == ["c.jpg", "b.jpg", "a.jpg"]
    assert library.current_item is current  # sort_items re-pointed current_index at "b.jpg"
    assert library.current_index == 1
