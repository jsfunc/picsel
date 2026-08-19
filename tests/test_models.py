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


def test_load_missing_state_file_leaves_load_error_none(tmp_path):
    _make_image(tmp_path / "a.jpg")
    library = ImageLibrary()
    library.load(tmp_path)
    assert library.load_error is None


def test_load_corrupted_state_file_sets_load_error(tmp_path):
    _make_image(tmp_path / "a.jpg")
    (tmp_path / ".picsel_state.json").write_text("not valid json{{{")

    library = ImageLibrary()
    library.load(tmp_path)

    assert library.load_error is not None
    assert str(tmp_path / ".picsel_state.json") in library.load_error
    assert library.items[0].status is Status.UNRATED  # degraded to unrated, not crashed


def test_load_non_dict_state_file_sets_load_error_instead_of_raising(tmp_path):
    # A structurally-odd-but-valid JSON file (e.g. after a cloud-sync
    # conflict copy) must degrade gracefully, not crash folder-open with an
    # AttributeError from treating a list/null as a dict.
    _make_image(tmp_path / "a.jpg")
    (tmp_path / ".picsel_state.json").write_text("[]")

    library = ImageLibrary()
    library.load(tmp_path)  # must not raise

    assert library.load_error is not None
    assert library.items[0].status is Status.UNRATED


def test_load_error_resets_on_a_later_successful_load(tmp_path):
    (tmp_path / ".picsel_state.json").write_text("not valid json{{{")
    other = tmp_path / "other"
    other.mkdir()

    library = ImageLibrary()
    library.load(tmp_path)
    assert library.load_error is not None

    library.load(other)  # a folder with no state file at all -- perfectly normal
    assert library.load_error is None


def test_save_state_writes_atomically(tmp_path, monkeypatch):
    _make_image(tmp_path / "a.jpg")
    library = ImageLibrary()
    library.load(tmp_path)
    library.set_status(0, Status.SELECTED)
    library.save_state()
    original_bytes = (tmp_path / ".picsel_state.json").read_bytes()

    library.set_status(0, Status.REJECTED)

    # Writes garbage to whatever file is actually targeted, then fails --
    # simulating a real partial/failed write. This is the detail that
    # distinguishes "wrote to a temp file, so the original is untouched"
    # from "wrote straight into the original, so it's now corrupted": a
    # mock that merely raises without writing anything would trivially
    # "pass" even for a non-atomic implementation, since no bytes would
    # ever land anywhere either way.
    real_write_bytes = Path.write_bytes

    def failing_write_bytes(self, data):
        real_write_bytes(self, b"CORRUPTED-PARTIAL-WRITE")
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", failing_write_bytes)

    with pytest.raises(OSError):
        library.save_state()

    # The original must be untouched -- save_state() writes to a temp file
    # first and only replaces the original once the write fully succeeds.
    assert (tmp_path / ".picsel_state.json").read_bytes() == original_bytes


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
