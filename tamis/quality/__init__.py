"""Aesthetic quality scoring (optional feature).

`store` is dependency-free and always importable; `scorer` needs the optional
`open_clip_torch` extra (see requirements-quality.txt) and is imported lazily
by the worker, so a build without it still loads this package.
"""

from tamis.quality.store import QUALITY_FILENAME, QualityStore

__all__ = ["QUALITY_FILENAME", "QualityStore"]
