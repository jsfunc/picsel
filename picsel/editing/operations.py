"""Pillow-based image editing operations and an undo/redo-aware edit session."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

from picsel.io_ops.file_ops import unique_path

NEUTRAL_ADJUSTMENTS = {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0}


def rotate90(image: Image.Image, clockwise: bool = True) -> Image.Image:
    transpose = Image.Transpose.ROTATE_270 if clockwise else Image.Transpose.ROTATE_90
    return image.transpose(transpose)


def flip_horizontal(image: Image.Image) -> Image.Image:
    return ImageOps.mirror(image)


def flip_vertical(image: Image.Image) -> Image.Image:
    return ImageOps.flip(image)


def crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    left = max(0, min(left, image.width))
    top = max(0, min(top, image.height))
    right = max(left, min(right, image.width))
    bottom = max(top, min(bottom, image.height))
    return image.crop((left, top, right, bottom))


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def adjust_saturation(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(image).enhance(factor)


def resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)


@dataclass
class Op:
    name: str
    params: dict = field(default_factory=dict)


class EditSession:
    """Tracks an original image plus an undo/redo stack of edit operations."""

    def __init__(self, image: Image.Image, source_path: Path | None = None) -> None:
        self.source_path = source_path
        self._original = ImageOps.exif_transpose(image).convert("RGB")
        self._ops: list[Op] = []
        self._redo_stack: list[Op] = []
        self._adjustments: dict[str, float] = dict(NEUTRAL_ADJUSTMENTS)
        # Snapshot of _ops/_adjustments as of the last successful save, so has_edits()
        # reflects edits not yet persisted anywhere rather than just "any edits at all".
        self._saved_ops: list[Op] = []
        self._saved_adjustments: dict[str, float] = dict(NEUTRAL_ADJUSTMENTS)

    @classmethod
    def from_path(cls, path: Path) -> EditSession:
        path = Path(path)
        with Image.open(path) as img:
            img.load()
            return cls(img, source_path=path)

    @staticmethod
    def _apply_op(image: Image.Image, op: Op) -> Image.Image:
        if op.name == "rotate90":
            return rotate90(image, op.params["clockwise"])
        if op.name == "flip_horizontal":
            return flip_horizontal(image)
        if op.name == "flip_vertical":
            return flip_vertical(image)
        if op.name == "crop":
            return crop(image, op.params["box"])
        if op.name == "resize":
            return resize(image, op.params["size"])
        if op.name == "adjust":
            image = adjust_brightness(image, op.params["brightness"])
            image = adjust_contrast(image, op.params["contrast"])
            image = adjust_saturation(image, op.params["saturation"])
            return image
        raise ValueError(f"Unknown op: {op.name}")

    def render(self) -> Image.Image:
        image = self._original
        for op in self._ops:
            image = self._apply_op(image, op)
        if self._adjustments != NEUTRAL_ADJUSTMENTS:
            image = self._apply_op(image, Op("adjust", dict(self._adjustments)))
        return image

    def _push_op(self, op: Op) -> None:
        self._ops.append(op)
        self._redo_stack.clear()

    def rotate(self, clockwise: bool = True) -> None:
        self._push_op(Op("rotate90", {"clockwise": clockwise}))

    def flip_horizontal(self) -> None:
        self._push_op(Op("flip_horizontal"))

    def flip_vertical(self) -> None:
        self._push_op(Op("flip_vertical"))

    def crop(self, box: tuple[int, int, int, int]) -> None:
        self._push_op(Op("crop", {"box": tuple(box)}))

    def resize(self, size: tuple[int, int]) -> None:
        self._push_op(Op("resize", {"size": tuple(size)}))

    def set_adjustments(self, brightness: float = 1.0, contrast: float = 1.0, saturation: float = 1.0) -> None:
        """Set live preview adjustment factors (not yet part of the undo stack)."""
        self._adjustments = {"brightness": brightness, "contrast": contrast, "saturation": saturation}

    def commit_adjustments(self) -> None:
        """Bake the current adjustment factors into the undo stack as one op."""
        if self._adjustments != NEUTRAL_ADJUSTMENTS:
            self._push_op(Op("adjust", dict(self._adjustments)))
            self._adjustments = dict(NEUTRAL_ADJUSTMENTS)

    def undo(self) -> None:
        if self._ops:
            self._redo_stack.append(self._ops.pop())

    def redo(self) -> None:
        if self._redo_stack:
            self._ops.append(self._redo_stack.pop())

    def can_undo(self) -> bool:
        return bool(self._ops)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def reset(self) -> None:
        self._ops.clear()
        self._redo_stack.clear()
        self._adjustments = dict(NEUTRAL_ADJUSTMENTS)

    def has_edits(self) -> bool:
        return self._ops != self._saved_ops or self._adjustments != self._saved_adjustments

    def save(self, path: Path | None = None, overwrite: bool = False) -> Path:
        image = self.render()
        if path is None:
            if self.source_path is None:
                raise ValueError("No source path available; pass an explicit path")
            if overwrite:
                path = self.source_path
            else:
                edited = self.source_path.with_name(
                    f"{self.source_path.stem}_edited{self.source_path.suffix}"
                )
                path = unique_path(edited)

        save_kwargs = {}
        if path.suffix.lower() in (".jpg", ".jpeg"):
            save_kwargs["quality"] = 95
        # Read exif from `_original`, not the rendered image: ImageEnhance (used for
        # brightness/contrast/saturation) drops `.info` entirely, so a rendered image
        # that went through an adjustment would otherwise report no exif at all.
        # Re-serialized via getexif().tobytes() rather than passed through as the
        # raw `_original.info["exif"]` blob: a real camera JPEG's raw blob can carry
        # an embedded IFD1 thumbnail depicting the *original* framing, and Pillow's
        # JPEG writer copies a raw bytes `exif=` blob through completely unparsed --
        # so a crop meant to remove someone/something from the photo could otherwise
        # leave them fully visible in the saved file's embedded thumbnail. Exif.tobytes()
        # only ever serializes IFD0 plus explicitly-accessed sub-IFDs (Exif/GPS/Interop),
        # never IFD1, so this can't happen.
        exif = self._original.getexif()
        if exif:
            save_kwargs["exif"] = exif.tobytes()

        # Write to a same-directory temp file and rename into place, rather than
        # letting Pillow's JPEG writer stream straight into `path` -- when path is
        # the original being overwritten, a crash or a full disk mid-write would
        # otherwise leave the user's original photo truncated with no way back.
        # os.replace() is atomic on both POSIX and Windows as long as source and
        # destination are on the same filesystem, which a same-directory temp file
        # guarantees.
        temp_path = path.with_name(f".picsel_save_{uuid.uuid4().hex}{path.suffix}")
        try:
            image.save(temp_path, **save_kwargs)
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        self._saved_ops = list(self._ops)
        self._saved_adjustments = dict(self._adjustments)
        return path
