"""Data model for a single image in a library."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Status(Enum):
    UNRATED = "unrated"
    SELECTED = "selected"
    REJECTED = "rejected"


@dataclass
class ImageItem:
    path: Path
    status: Status = Status.UNRATED
    rating: int = 0

    @property
    def name(self) -> str:
        return self.path.name
