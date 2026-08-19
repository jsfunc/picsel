from pathlib import Path

import pytest
from PIL import Image

from picsel.thumbnails import generate_thumbnail, load_qimage, pil_to_qimage


def _make_image(path: Path, size=(40, 30), color=(10, 20, 30)) -> None:
    Image.new("RGB", size, color).save(path)


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
