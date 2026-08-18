from pathlib import Path

from PIL import Image

from picsel.editing import EditSession


def _make_session(tmp_path: Path, size=(8, 6), color=(10, 20, 30)) -> tuple[EditSession, Path]:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", size, color).save(path)
    return EditSession.from_path(path), path


def test_rotate_swaps_dimensions(tmp_path):
    session, _ = _make_session(tmp_path, size=(8, 6))
    session.rotate(clockwise=True)
    image = session.render()
    assert (image.width, image.height) == (6, 8)


def test_undo_redo(tmp_path):
    session, _ = _make_session(tmp_path)
    assert not session.can_undo()

    session.rotate()
    assert session.can_undo()
    assert not session.can_redo()
    rotated_size = session.render().size

    session.undo()
    assert not session.can_undo()
    assert session.can_redo()
    assert session.render().size != rotated_size

    session.redo()
    assert session.render().size == rotated_size


def test_crop(tmp_path):
    session, _ = _make_session(tmp_path, size=(10, 10))
    session.crop((2, 2, 6, 8))
    assert session.render().size == (4, 6)


def test_crop_clamps_out_of_bounds_box(tmp_path):
    session, _ = _make_session(tmp_path, size=(10, 10))
    session.crop((-5, -5, 100, 100))
    assert session.render().size == (10, 10)


def test_adjustments_are_not_committed_until_commit_adjustments(tmp_path):
    session, _ = _make_session(tmp_path)
    session.set_adjustments(brightness=1.5, contrast=1.0, saturation=1.0)
    assert session.has_edits()
    assert not session.can_undo()  # live preview only, not yet on the undo stack

    session.commit_adjustments()
    assert session.can_undo()
    assert session.has_edits()  # now tracked as a committed op rather than a live preview


def test_reset_clears_ops_and_adjustments(tmp_path):
    session, _ = _make_session(tmp_path)
    session.rotate()
    session.set_adjustments(brightness=1.5)
    session.reset()
    assert not session.has_edits()
    assert not session.can_undo()


def test_save_preserves_exif(tmp_path):
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    exif = Image.Exif()
    exif[0x9003] = "2020:01:01 12:00:00"  # DateTimeOriginal
    image.save(path, exif=exif.tobytes())

    session = EditSession.from_path(path)
    session.rotate()
    saved_path = session.save(overwrite=False)

    with Image.open(saved_path) as saved:
        saved_exif = saved.getexif()
    assert saved_exif.get(0x9003) == "2020:01:01 12:00:00"


def test_save_preserves_exif_after_adjustments(tmp_path):
    # ImageEnhance (used for brightness/contrast/saturation) drops PIL's `.info`
    # dict entirely, so exif must be read from the original image, not the
    # rendered one, or an adjusted save silently loses all EXIF data.
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    exif = Image.Exif()
    exif[0x9003] = "2020:01:01 12:00:00"  # DateTimeOriginal
    image.save(path, exif=exif.tobytes())

    session = EditSession.from_path(path)
    session.set_adjustments(brightness=1.2, contrast=1.0, saturation=1.0)
    session.commit_adjustments()
    saved_path = session.save(overwrite=False)

    with Image.open(saved_path) as saved:
        saved_exif = saved.getexif()
    assert saved_exif.get(0x9003) == "2020:01:01 12:00:00"


def test_save_copy_does_not_overwrite_original(tmp_path):
    session, path = _make_session(tmp_path)
    session.rotate()
    saved_path = session.save(overwrite=False)
    assert saved_path != path
    assert saved_path.exists()
    assert path.exists()


def test_save_overwrite_replaces_original(tmp_path):
    session, path = _make_session(tmp_path)
    session.rotate()
    saved_path = session.save(overwrite=True)
    assert saved_path == path
