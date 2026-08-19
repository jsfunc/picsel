"""Thumbnail and full-image loading, including async workers for use with QThreadPool."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QImage

from picsel.metadata import MetadataSection, extract_metadata

logger = logging.getLogger(__name__)

DEFAULT_THUMBNAIL_SIZE = (160, 160)

# Pillow's default (~89.5 megapixels) exists to guard against maliciously
# crafted files designed to exhaust memory when decoded -- a small file
# whose header claims an enormous pixel count. picSel only ever opens files
# already on the user's own disk, and genuinely large real photos are
# explicitly in scope (100MP+ sensors, stitched panoramas, high-DPI scans),
# so the stock limit is too tight for legitimate use. Raised well past any
# real camera/scanner output, but still bounded -- an image past this is
# almost certainly a corrupt or adversarial file, not a real photo, and
# it's still large enough that decoding it could stress an ordinary
# machine's memory, so it's worth refusing with a clear reason rather than
# either silently failing or risking a multi-gigabyte allocation attempt.
Image.MAX_IMAGE_PIXELS = 300_000_000


def pil_to_qimage(image: Image.Image) -> QImage:
    """Convert a PIL image to an owned QImage (RGB888)."""
    image = image.convert("RGB")
    data = image.tobytes("raw", "RGB")
    qimage = QImage(data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888)
    return qimage.copy()


def _reraise_decompression_bomb_clearly(path: Path, exc: Image.DecompressionBombError) -> None:
    # Pillow's own message ends with "...could be decompression bomb DOS
    # attack" -- alarming and misleading for what's almost always just a
    # very large but legitimate photo. Lead with a plain explanation and
    # keep Pillow's own numbers (it already reports the actual pixel count
    # and the configured limit) rather than re-deriving them.
    raise OSError(f"{path.name} is unusually large and was skipped for safety ({exc})") from exc


def generate_thumbnail(path: Path, size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE) -> QImage:
    """Decode an image file and return a thumbnail-sized QImage (RGB888)."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail(size)
            return pil_to_qimage(img)
    except Image.DecompressionBombError as exc:
        _reraise_decompression_bomb_clearly(path, exc)


def load_qimage(path: Path) -> QImage:
    """Decode an image file at full resolution, applying EXIF orientation."""
    try:
        with Image.open(path) as img:
            img.load()
            img = ImageOps.exif_transpose(img)
            return pil_to_qimage(img)
    except Image.DecompressionBombError as exc:
        _reraise_decompression_bomb_clearly(path, exc)


class ThumbnailSignals(QObject):
    finished = Signal(object, QImage, str)  # path, image (null on failure), error message ("" if ok)


class ThumbnailWorker(QRunnable):
    """Generates a thumbnail off the UI thread and emits the result via signals."""

    def __init__(self, path: Path, size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE) -> None:
        super().__init__()
        self.path = path
        self.size = size
        self.signals = ThumbnailSignals()

    def run(self) -> None:
        try:
            image = generate_thumbnail(self.path, self.size)
            error = ""
        except Exception as exc:
            logger.warning("Failed to generate thumbnail for %s: %s", self.path, exc)
            image = QImage()
            error = str(exc)
        self.signals.finished.emit(self.path, image, error)


class ImageLoadSignals(QObject):
    finished = Signal(object, QImage, str)  # path, image (null on failure), error message ("" if ok)


class ImageLoadWorker(QRunnable):
    """Decodes a full-resolution image off the UI thread and emits the result via signals."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.signals = ImageLoadSignals()

    def run(self) -> None:
        try:
            image = load_qimage(self.path)
            error = ""
        except Exception as exc:
            image = QImage()
            error = str(exc)
        self.signals.finished.emit(self.path, image, error)


class MetadataLoadSignals(QObject):
    finished = Signal(object, list, str)  # path, list[MetadataSection] (empty on failure), error message ("" if ok)


class MetadataLoadWorker(QRunnable):
    """Reads EXIF/GPS metadata off the UI thread and emits the result via
    signals -- extract_metadata() does file I/O plus IFD parsing per call,
    which previously ran synchronously on the UI thread on every single
    photo navigation, the one exception to this app's otherwise-consistent
    "decode/detect off the UI thread" pattern (thumbnails, full-image
    decode, and face detection are all already async via this same
    QThreadPool)."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.signals = MetadataLoadSignals()

    def run(self) -> None:
        try:
            sections: list[MetadataSection] = extract_metadata(self.path)
            error = ""
        except Exception as exc:
            sections = []
            error = str(exc)
        self.signals.finished.emit(self.path, sections, error)
