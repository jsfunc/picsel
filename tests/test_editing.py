from pathlib import Path

import pytest
from PIL import Image

from tamis.editing import EditSession


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


def test_save_preserves_exif_sub_ifd_data(tmp_path):
    # Regression test: real camera metadata -- DateTimeOriginal, GPS
    # coordinates, exposure/ISO/lens info -- lives in the Exif and GPS
    # sub-IFDs (tags 0x8769/0x8825), not IFD0. Exif.tobytes() only
    # serializes a sub-IFD that's actually been accessed via get_ifd(), so
    # save() explicitly touches them first. Requires Pillow >=11.1 to
    # actually pass, though (see operations.py's comment) -- confirmed
    # empirically that get_ifd() itself fails to retrieve a freshly-loaded
    # Exif object's sub-IFD contents at all on older versions, which no
    # amount of explicit touching works around; requirements.txt pins the
    # floor there for exactly this reason.
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    exif = Image.Exif()
    exif[0x010F] = "RealCam"  # Make -- IFD0
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = "2021:07:04 08:15:00"  # DateTimeOriginal -- Exif sub-IFD
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"  # GPSLatitudeRef -- GPS sub-IFD
    image.save(path, exif=exif.tobytes())

    session = EditSession.from_path(path)
    session.rotate()
    saved_path = session.save(overwrite=False)

    with Image.open(saved_path) as saved:
        saved_exif = saved.getexif()
        assert saved_exif.get(0x010F) == "RealCam"
        saved_sub = dict(saved_exif.get_ifd(0x8769))
        assert saved_sub.get(0x9003) == "2021:07:04 08:15:00", saved_sub
        saved_gps = dict(saved_exif.get_ifd(0x8825))
        assert saved_gps.get(1) == "N", saved_gps


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


def test_save_does_not_pass_the_raw_exif_blob_through_verbatim(tmp_path):
    # A real camera JPEG's raw exif blob can carry an embedded IFD1
    # thumbnail depicting the original framing. Passing that raw blob
    # through unparsed to Pillow's JPEG writer would leave a cropped-out
    # subject fully visible in the saved file's embedded thumbnail even
    # though the real pixels are correctly cropped -- save() must instead
    # re-serialize via getexif().tobytes(), which only ever includes IFD0
    # plus explicitly-accessed sub-IFDs, never a stale IFD1/thumbnail.
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (40, 40), (10, 20, 30))
    exif = Image.Exif()
    exif[0x9003] = "2020:01:01 12:00:00"  # DateTimeOriginal
    image.save(path, exif=exif.tobytes())

    session = EditSession.from_path(path)
    # Simulate a real camera file's raw exif blob carrying extra payload
    # beyond plain IFD0 tags (standing in for an embedded IFD1 thumbnail).
    # getexif() is unaffected by this since it was parsed once at load time
    # -- exactly the property that makes save() safe.
    sentinel = b"FAKE-THUMBNAIL-PAYLOAD-SENTINEL"
    session._original.info["exif"] = session._original.info["exif"] + sentinel

    session.crop((5, 5, 35, 35))
    saved_path = session.save(overwrite=False)

    saved_bytes = saved_path.read_bytes()
    assert sentinel not in saved_bytes
    with Image.open(saved_path) as saved:
        assert saved.getexif().get(0x9003) == "2020:01:01 12:00:00"  # real tags still preserved


def test_save_overwrite_does_not_corrupt_original_if_the_write_fails(tmp_path, monkeypatch):
    session, path = _make_session(tmp_path)
    original_bytes = path.read_bytes()
    session.rotate()

    # Writes garbage to whatever file Pillow was actually asked to save to,
    # then fails -- simulating a real partial/failed write (e.g. disk full
    # partway through), not just an error before any bytes are touched.
    # This is the detail that distinguishes "wrote to a temp file, so the
    # original is untouched" from "wrote straight into the original, so it's
    # now corrupted": if save() ever regresses to writing `path` directly,
    # this test would find `path` itself holding the garbage bytes below.
    def failing_save(self, fp, *args, **kwargs):
        Path(fp).write_bytes(b"CORRUPTED-PARTIAL-WRITE")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", failing_save)

    with pytest.raises(OSError):
        session.save(overwrite=True)

    # The original must be untouched -- save() writes to a temp file first
    # and only replaces the original once the write fully succeeds.
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".tamis_save_*"))  # no leftover temp file either
