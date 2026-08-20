from pathlib import Path

import pytest
from PIL import Image

from tamis.thumbnails import generate_thumbnail, load_qimage, pil_to_qimage


def _make_image(path: Path, size=(40, 30), color=(10, 20, 30)) -> None:
    Image.new("RGB", size, color).save(path)


def _make_image_with_orientation(path: Path, size, orientation: int, color=(200, 50, 50)) -> None:
    img = Image.new("RGB", size, color)
    exif = Image.Exif()
    exif[0x0112] = orientation  # Orientation tag
    img.save(path, exif=exif.tobytes())


def test_generate_thumbnail_produces_a_scaled_down_image(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path, size=(400, 300))

    thumb = generate_thumbnail(path, size=(160, 160))

    assert thumb.width() <= 160
    assert thumb.height() <= 160


def test_load_qimage_preserves_full_resolution(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path, size=(400, 300))

    image = load_qimage(path)

    assert (image.width(), image.height()) == (400, 300)


def test_pil_to_qimage_round_trips_pixel_colors(tmp_path):
    img = Image.new("RGB", (2, 2), (10, 20, 30))
    qimage = pil_to_qimage(img)
    assert qimage.pixelColor(0, 0).getRgb()[:3] == (10, 20, 30)


def test_generate_thumbnail_raises_a_clear_message_for_oversized_images(tmp_path, monkeypatch):
    # Regression test: an oversized image previously surfaced only as a
    # generic Pillow error ("...could be decompression bomb DOS attack"),
    # alarming and unhelpful for what's almost always just a large but
    # legitimate photo. A tiny MAX_IMAGE_PIXELS makes even a small test
    # image "exceed" the limit, without needing to actually allocate a huge
    # real image to exercise the failure path.
    path = tmp_path / "huge.jpg"
    _make_image(path, size=(100, 100))

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(OSError) as exc_info:
        generate_thumbnail(path)

    message = str(exc_info.value)
    assert "unusually large" in message
    assert "huge.jpg" in message


def test_load_qimage_raises_a_clear_message_for_oversized_images(tmp_path, monkeypatch):
    path = tmp_path / "huge.jpg"
    _make_image(path, size=(100, 100))

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(OSError) as exc_info:
        load_qimage(path)

    message = str(exc_info.value)
    assert "unusually large" in message
    assert "huge.jpg" in message


def test_generate_thumbnail_still_applies_exif_orientation_correctly(tmp_path):
    # Regression test for downscale-before-transpose: must still produce a
    # correctly-oriented result. Orientation 6 means the raw pixel data
    # (400 wide x 300 tall, landscape) should display rotated 90 degrees --
    # i.e. as a 300x400 portrait image.
    path = tmp_path / "photo.jpg"
    _make_image_with_orientation(path, size=(400, 300), orientation=6)

    thumb = generate_thumbnail(path, size=(160, 160))

    assert thumb.height() > thumb.width()


def test_generate_thumbnail_uses_draft_mode_for_jpegs(tmp_path, monkeypatch):
    # Regression test for the actual performance claim, not just that the
    # code still runs: Image.thumbnail() only engages JPEG draft mode (a
    # fast, approximate decode that skips most of the real pixel work) if
    # the image hasn't been fully decoded yet. exif_transpose() forces a
    # full load() as its very first step, so it must run *after*
    # thumbnail(), not before, for draft mode to ever actually fire.
    path = tmp_path / "photo.jpg"
    _make_image(path, size=(800, 600))

    from PIL import JpegImagePlugin

    # JpegImageFile defines its own draft() (overriding Image.Image.draft),
    # which is what self.draft(...) actually resolves to for a real JPEG --
    # patching the base class method wouldn't intercept anything.
    draft_calls = []
    real_draft = JpegImagePlugin.JpegImageFile.draft

    def spying_draft(self, mode, size):
        result = real_draft(self, mode, size)
        draft_calls.append(result)
        return result

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spying_draft)

    generate_thumbnail(path, size=(160, 160))

    assert draft_calls, "Image.draft() was never called -- draft mode isn't engaging"
    assert draft_calls[0] is not None, "draft() ran but declined to do anything (image was already loaded?)"
