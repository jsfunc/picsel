"""Sharpness scoring: variance of the Laplacian, mapped to 0-100.

No torch and no Qt -- this is arithmetic over a decoded image, ~1ms, and it
rides along on a decode the caller has already paid for.

What it measures, and what it does not. Laplacian variance is edge energy, so
it answers "how much fine detail is present", which is a good proxy for focus
and a poor one for content: a sharp photo of a plain wall or an empty sky has
little edge energy and scores low, while a busy but slightly soft scene can
score higher than a clean, crisply-focused one. Read a low score as "worth
looking at", not as a verdict.

Resolution matters more than it looks. Measured against full-resolution values
over 150 photos, the rank correlation is 0.92 when the image is decoded via
JPEG draft mode at roughly 500px, but drops to 0.72 once the image has been
*resampled* down to a 160px thumbnail -- resampling is itself a low-pass
filter, so it erases the very differences being measured. Score from the
draft-decoded image, never from a finished thumbnail.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# The 0-100 window, as raw Laplacian variance. Chosen from the distribution of
# a 455-photo unculled folder (min 2.5, p10 295, median 1448, max 8114): this
# window leaves nothing clamped at the bottom and only ~7% at the top, while
# putting the genuinely blurred frames below 20.
#
# Deliberately absolute rather than a percentile within the folder, so the
# number means the same thing in every folder. A percentile would force 10% of
# any folder below 10 even when every photo in it is sharp.
RAW_MIN, RAW_MAX = 2.0, 4000.0
_LOG_MIN, _LOG_MAX = float(np.log10(RAW_MIN)), float(np.log10(RAW_MAX))


def laplacian_variance(image: Image.Image) -> float:
    """Raw edge energy of `image`. Higher is sharper."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    # 4-neighbour Laplacian written out rather than convolved: it is one
    # expression, needs no scipy, and runs in about a millisecond.
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def blur_score(image: Image.Image) -> int:
    """Sharpness of `image` as 0-100, where low means blurred.

    Log-scaled, because sharpness spans three orders of magnitude across a
    real folder (2.5 to 8114) and a linear map would put every in-focus photo
    within a point or two of 100.
    """
    raw = laplacian_variance(image)
    if raw <= 0:
        return 0
    fraction = (float(np.log10(raw)) - _LOG_MIN) / (_LOG_MAX - _LOG_MIN)
    return int(round(float(np.clip(fraction, 0.0, 1.0)) * 100))
