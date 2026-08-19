import os
import time
from pathlib import Path

import pytest
from PIL import Image

from picsel.io_ops import (
    apply_culling,
    capture_time,
    find_sequence_groups,
    rename_by_creation_date,
    renumber_by_creation_time,
    rename_with_sequence,
    unique_path,
)
from picsel.models import ImageItem, ImageLibrary, Status


def _make_image(path: Path) -> None:
    Image.new("RGB", (4, 4), (0, 0, 0)).save(path)


def _make_image_with_capture_time(path: Path, when: float) -> None:
    """Save a JPEG with a real EXIF DateTimeOriginal, in the sub-IFD like a camera
    actually writes it (not flattened into IFD0)."""
    import datetime as _dt

    exif = Image.Exif()
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = _dt.datetime.fromtimestamp(when).strftime("%Y:%m:%d %H:%M:%S")
    Image.new("RGB", (4, 4), (0, 0, 0)).save(path, exif=exif.tobytes())


def test_unique_path_returns_same_path_if_free(tmp_path):
    target = tmp_path / "photo.jpg"
    assert unique_path(target) == target


def test_unique_path_appends_counter_on_collision(tmp_path):
    target = tmp_path / "photo.jpg"
    _make_image(target)
    result = unique_path(target)
    assert result == tmp_path / "photo (1).jpg"

    _make_image(result)
    result2 = unique_path(target)
    assert result2 == tmp_path / "photo (2).jpg"


def test_rename_with_sequence(tmp_path):
    path = tmp_path / "IMG_0001.jpg"
    _make_image(path)
    item = ImageItem(path=path)

    new_path = rename_with_sequence(item, "trip", 3)

    assert new_path.name == "trip003.jpg"
    assert new_path.exists()
    assert not path.exists()
    assert item.path == new_path


def test_rename_with_sequence_handles_case_only_rename_on_case_insensitive_filesystem(tmp_path, monkeypatch):
    # This machine's filesystem is case-sensitive, so simulate a
    # case-insensitive one (default macOS/Windows) by making Path.exists()
    # match case-insensitively against whatever's actually on disk --
    # exactly the behavior that made a naive unique_path(destination) check
    # see the file being renamed as "already existing" under its new
    # casing, appending " (1)" instead of doing the plain rename.
    path = tmp_path / "photo001.jpg"
    _make_image(path)
    item = ImageItem(path=path)

    real_exists = Path.exists

    def case_insensitive_exists(self, *args, **kwargs):
        if real_exists(self, *args, **kwargs):
            return True
        try:
            names_on_disk = {p.name.lower() for p in self.parent.iterdir()}
        except OSError:
            return False
        return self.name.lower() in names_on_disk

    monkeypatch.setattr(Path, "exists", case_insensitive_exists)

    new_path = rename_with_sequence(item, "PHOTO", 1)

    assert new_path.name == "PHOTO001.jpg"  # not "PHOTO001 (1).jpg"
    assert item.path == new_path
    assert new_path.exists()


def test_apply_culling_moves_selected_and_rejected(tmp_path):
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "b.jpg")
    _make_image(tmp_path / "c.jpg")

    library = ImageLibrary()
    library.load(tmp_path)
    library.set_status(0, Status.SELECTED)
    library.set_status(1, Status.REJECTED)
    # c.jpg left unrated

    report = apply_culling(library, mode="move", selected_dir="selected", rejected_dir="rejected")

    assert report.moved_selected == 1
    assert report.moved_rejected == 1
    assert not report.errors
    assert (tmp_path / "selected" / "a.jpg").exists()
    assert (tmp_path / "rejected" / "b.jpg").exists()
    assert (tmp_path / "c.jpg").exists()
    assert not (tmp_path / "a.jpg").exists()


def test_apply_culling_copy_mode_leaves_originals(tmp_path):
    _make_image(tmp_path / "a.jpg")
    library = ImageLibrary()
    library.load(tmp_path)
    library.set_status(0, Status.SELECTED)

    apply_culling(library, mode="copy", selected_dir="selected", rejected_dir="rejected")

    assert (tmp_path / "a.jpg").exists()
    assert (tmp_path / "selected" / "a.jpg").exists()


def test_apply_culling_cleans_up_a_partial_write_on_failure(tmp_path, monkeypatch):
    # Regression test: a copy interrupted mid-write (disk full, a dropped
    # network mount) previously left a truncated file at the destination.
    # A retry would then see that leftover as "already there" and create a
    # differently-named duplicate right next to the corruption instead of
    # detecting or clearing it.
    import shutil

    from picsel.io_ops import file_ops

    _make_image(tmp_path / "a.jpg")
    library = ImageLibrary()
    library.load(tmp_path)
    library.set_status(0, Status.SELECTED)

    real_copy2 = shutil.copy2

    def failing_copy2(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"CORRUPTED-PARTIAL-WRITE")
        raise OSError("disk full")

    monkeypatch.setattr(file_ops.shutil, "copy2", failing_copy2)

    report = apply_culling(library, mode="copy", selected_dir="selected", rejected_dir="rejected")

    assert report.errors
    assert not (tmp_path / "selected" / "a.jpg").exists(), "the partial/corrupted file was not cleaned up"

    # A retry (once whatever caused the failure is fixed) should land on the
    # plain expected filename, not a "(1)" duplicate next to leftover junk.
    monkeypatch.setattr(file_ops.shutil, "copy2", real_copy2)
    library.set_status(0, Status.SELECTED)
    report2 = apply_culling(library, mode="copy", selected_dir="selected", rejected_dir="rejected")
    assert not report2.errors
    assert (tmp_path / "selected" / "a.jpg").exists()
    assert not (tmp_path / "selected" / "a (1).jpg").exists()


def test_apply_culling_requires_folder():
    library = ImageLibrary()
    with pytest.raises(ValueError):
        apply_culling(library)


def test_apply_culling_rejects_invalid_mode(tmp_path):
    library = ImageLibrary()
    library.load(tmp_path)
    with pytest.raises(ValueError):
        apply_culling(library, mode="bogus")


def test_find_sequence_groups_ignores_singletons_and_non_sequenced_names(tmp_path):
    _make_image(tmp_path / "toto001.jpg")
    _make_image(tmp_path / "toto002.jpg")
    _make_image(tmp_path / "toto003.jpg")
    _make_image(tmp_path / "solo005.jpg")  # only member of its group
    _make_image(tmp_path / "notes.jpg")  # no trailing digits at all

    library = ImageLibrary()
    library.load(tmp_path)
    groups = find_sequence_groups(library.items)

    assert set(groups.keys()) == {"toto"}
    assert {item.name for item in groups["toto"]} == {"toto001.jpg", "toto002.jpg", "toto003.jpg"}


def test_renumber_by_creation_time_swaps_out_of_order_files(tmp_path):
    # toto001.jpg was actually created AFTER toto002.jpg -> renumbering should swap them.
    now = time.time()
    path1 = tmp_path / "toto001.jpg"
    path2 = tmp_path / "toto002.jpg"
    _make_image(path1)
    _make_image(path2)
    os.utime(path2, (now, now))
    os.utime(path1, (now + 100, now + 100))

    library = ImageLibrary()
    library.load(tmp_path)
    item_originally_001 = next(item for item in library.items if item.path == path1)
    item_originally_002 = next(item for item in library.items if item.path == path2)

    report = renumber_by_creation_time(library.items, "toto")

    assert report.renamed == 2
    assert not report.errors
    # toto002 (created earlier) should now be first; toto001 (created later) now second.
    assert item_originally_002.path.name == "toto001.jpg"
    assert item_originally_001.path.name == "toto002.jpg"


def test_renumber_by_creation_time_uses_exif_over_mtime(tmp_path):
    path1 = tmp_path / "toto001.jpg"
    path2 = tmp_path / "toto002.jpg"
    # EXIF says toto001 was taken first, but mtimes deliberately say the opposite,
    # to prove EXIF capture time wins over filesystem mtime.
    _make_image_with_capture_time(path1, when=1_000_000)
    _make_image_with_capture_time(path2, when=1_000_100)
    now = time.time()
    os.utime(path1, (now + 100, now + 100))
    os.utime(path2, (now, now))

    library = ImageLibrary()
    library.load(tmp_path)
    original_item_for_path1 = next(item for item in library.items if item.path == path1)
    renumber_by_creation_time(library.items, "toto")

    assert original_item_for_path1.path.name == "toto001.jpg"


def test_renumber_no_op_for_single_match(tmp_path):
    _make_image(tmp_path / "toto001.jpg")
    library = ImageLibrary()
    library.load(tmp_path)
    report = renumber_by_creation_time(library.items, "toto")
    assert report.renamed == 0
    assert library.items[0].path.name == "toto001.jpg"


def test_rename_by_creation_date_uses_exif_over_mtime(tmp_path):
    import datetime as _dt

    path = tmp_path / "IMG_0001.jpg"
    when = _dt.datetime(2024, 3, 15, 14, 30, 22).timestamp()
    _make_image_with_capture_time(path, when=when)
    now = time.time()
    os.utime(path, (now, now))  # mtime deliberately different from EXIF

    library = ImageLibrary()
    library.load(tmp_path)

    report = rename_by_creation_date(library.items)

    assert report.renamed == 1
    assert not report.errors
    assert library.items[0].path.name == "p20240315_143022.jpg"
    assert library.items[0].path.exists()
    assert not path.exists()


def test_rename_by_creation_date_falls_back_to_mtime(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)  # no EXIF
    when = time.mktime((2023, 1, 2, 3, 4, 5, 0, 0, -1))
    os.utime(path, (when, when))

    library = ImageLibrary()
    library.load(tmp_path)

    report = rename_by_creation_date(library.items)

    assert report.renamed == 1
    assert library.items[0].path.name == "p20230102_030405.jpg"


def test_rename_by_creation_date_disambiguates_same_second_collisions(tmp_path):
    when = time.time()
    path1 = tmp_path / "a.jpg"
    path2 = tmp_path / "b.jpg"
    _make_image(path1)
    _make_image(path2)
    os.utime(path1, (when, when))
    os.utime(path2, (when, when))

    library = ImageLibrary()
    library.load(tmp_path)

    report = rename_by_creation_date(library.items)

    assert report.renamed == 2
    names = {item.path.name for item in library.items}
    stamp = time.strftime("p%Y%m%d_%H%M%S", time.localtime(when))
    assert names == {f"{stamp}.jpg", f"{stamp} (1).jpg"}


def test_rename_by_creation_date_preserves_original_extension_case(tmp_path):
    # Saved as a real JPEG then renamed to .HEIC: Pillow has no HEIC encoder
    # registered in tests, but rename_by_creation_date only cares about the
    # suffix on disk, not the actual image codec.
    jpg_path = tmp_path / "IMG_0002.jpg"
    _make_image(jpg_path)
    path = jpg_path.with_suffix(".HEIC")
    jpg_path.rename(path)
    when = time.time()
    os.utime(path, (when, when))

    library = ImageLibrary()
    library.load(tmp_path)

    report = rename_by_creation_date(library.items)

    assert report.renamed == 1
    assert library.items[0].path.suffix == ".HEIC"


def test_rename_by_creation_date_empty_library_is_a_no_op():
    report = rename_by_creation_date([])
    assert report.renamed == 0
    assert not report.errors


def test_sort_thumbnails_by_capture_time_not_just_name(tmp_path):
    # "b.jpg" was actually taken before "a.jpg" -- alphabetical (load) order
    # would show them in the wrong sequence; sorting by capture time should not.
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    _make_image_with_capture_time(path_a, when=2_000_000)
    _make_image_with_capture_time(path_b, when=1_000_000)

    library = ImageLibrary()
    library.load(tmp_path)
    assert [item.name for item in library.items] == ["a.jpg", "b.jpg"]  # name order

    library.sort_items(key=lambda item: capture_time(item.path))

    assert [item.name for item in library.items] == ["b.jpg", "a.jpg"]  # chronological order


def test_sort_by_stars_breaks_ties_by_capture_time(tmp_path):
    # y: 5 stars. x and z: both 3 stars, but x was taken before z. w: unrated.
    _make_image_with_capture_time(tmp_path / "w.jpg", when=500)
    _make_image_with_capture_time(tmp_path / "x.jpg", when=1_000)
    _make_image_with_capture_time(tmp_path / "y.jpg", when=2_000)
    _make_image_with_capture_time(tmp_path / "z.jpg", when=3_000)

    library = ImageLibrary()
    library.load(tmp_path)
    ratings = {"w.jpg": 0, "x.jpg": 3, "y.jpg": 5, "z.jpg": 3}
    for index, item in enumerate(library.items):
        library.set_rating(index, ratings[item.name])

    library.sort_items(key=lambda item: (-item.rating, capture_time(item.path)))

    # highest rating first (y=5); among the 3-star tie, earlier capture time
    # first (x before z); unrated (w) last.
    assert [item.name for item in library.items] == ["y.jpg", "x.jpg", "z.jpg", "w.jpg"]
