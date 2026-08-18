"""Face detection via a pretrained MTCNN. No app/Qt dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MTCNN is a 3-stage cascade (P-Net, R-Net, O-Net); by default each stage
# rejects candidates outright ([0.6, 0.7, 0.7]), so a genuine but partially
# occluded face can be dropped mid-cascade before it ever gets a final score
# -- observed in practice on a face mostly hidden by sunglasses/a cap, which
# scored 0.698 but was silently discarded at the default thresholds. Loosening
# the gates lets every candidate reach a final O-Net confidence instead, so
# `detect_faces` can return every candidate's real score rather than a
# pre-filtered list. This does surface a few more low-confidence false
# positives too, but none above the confidence false positives already reach
# at the default thresholds (e.g. a dry leaf scoring 0.974 -- see the
# recognition feature's design notes), so filtering by DEFAULT_MIN_CONFIDENCE
# (or any user-chosen cutoff) discards the new noise along with the rest.
_CASCADE_THRESHOLDS = [0.5, 0.5, 0.3]
DEFAULT_MIN_CONFIDENCE = 0.9

_mtcnn: MTCNN | None = None


def _get_mtcnn() -> MTCNN:
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=DEVICE, post_process=False, thresholds=_CASCADE_THRESHOLDS)
    return _mtcnn


@dataclass
class FaceDetection:
    box: tuple[int, int, int, int]  # left, top, right, bottom, in `image`'s own pixel coordinates
    confidence: float


def detect_faces(image: Image.Image) -> list[FaceDetection]:
    """Detect every face candidate in `image` (already EXIF-oriented and RGB-converted), unfiltered.

    Returns every candidate's real confidence score, deliberately unfiltered
    by any threshold -- run this once per image and cache the result, then
    apply a confidence cutoff (`DEFAULT_MIN_CONFIDENCE` or a user-chosen
    value) as a plain filter over the returned list wherever it's used (e.g.
    a live UI slider). That filtering is a free list comprehension, so it
    never needs to re-run detection.
    """
    boxes, probs = _get_mtcnn().detect(image)
    if boxes is None:
        return []
    return [
        FaceDetection(box=tuple(int(v) for v in box), confidence=float(prob)) for box, prob in zip(boxes, probs)
    ]


def load_for_detection(path: Path) -> Image.Image:
    """Load `path`, apply EXIF orientation, and convert to RGB, at full resolution.

    Deliberately matches `picsel.thumbnails.load_qimage` (also uncapped) so
    face boxes land in the exact same pixel coordinates as what the app
    actually displays and crops -- a downscaled detection pass would silently
    misplace box overlays and any user-drawn "add a face here" box handed to
    `embed_faces`. Full resolution costs more time (a few hundred ms on this
    machine's GPU for typical camera-resolution photos) but produced no new
    high-confidence false positives in testing versus a downscaled pass.
    """
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")
