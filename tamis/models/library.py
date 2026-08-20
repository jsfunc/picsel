"""Loads a folder of images and tracks culling state with sidecar persistence."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tamis.models.image_item import ImageItem, Status
from tamis.persistence import atomic_write_bytes

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".heic", ".heif",
}

STATE_FILENAME = ".tamis_state.json"
_LEGACY_STATE_FILENAME = ".picsel_state.json"  # pre-rename name (picSel -> Tamis); migrated automatically
RENAMED_NAMES_KEY = "_renamed_names"


class ImageLibrary:
    def __init__(self) -> None:
        self.folder: Path | None = None
        self.items: list[ImageItem] = []
        self.current_index: int = 0
        self.renamed_names: dict[str, int] = {}
        # Set by _load_state() only when this folder's state file *exists*
        # but couldn't be read -- as opposed to legitimately not existing
        # yet (a folder never opened in Tamis before). Mirrors
        # PersonGallery/FaceCatalog's load_error: without this, a corrupted
        # state file silently reset every photo's rating/status to
        # "unrated" with no warning, and the next save_state() would
        # permanently overwrite the (maybe still partially recoverable)
        # original file with that empty state.
        self.load_error: str | None = None

    def load(self, folder: Path) -> None:
        folder = Path(folder)
        # Resolve the listing before mutating any state, so a failed load (e.g.
        # unreadable folder) leaves the library exactly as it was rather than
        # pointing `self.folder` at a folder whose items were never loaded.
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        self.folder = folder
        self.current_index = 0
        self.renamed_names = {}
        self.items = [ImageItem(path=p) for p in files]
        self._load_state()

    def _state_path(self) -> Path:
        assert self.folder is not None
        return self.folder / STATE_FILENAME

    def _migrate_legacy_state_file(self) -> None:
        """One-time migration for a folder opened before the picSel -> Tamis
        rename: if the old sidecar exists and the new one doesn't yet, rename
        it in place so existing ratings/status keep working with no user
        action needed. Best-effort -- a failure here just means _load_state()
        starts fresh, same as if the folder had never been opened before."""
        assert self.folder is not None
        new_path = self._state_path()
        legacy_path = self.folder / _LEGACY_STATE_FILENAME
        if new_path.exists() or not legacy_path.exists():
            return
        try:
            legacy_path.rename(new_path)
        except OSError as exc:
            logger.warning("Could not migrate %s to %s: %s", legacy_path, new_path, exc)

    def _load_state(self) -> None:
        self.load_error = None
        self._migrate_legacy_state_file()
        state_path = self._state_path()
        if not state_path.exists():
            return
        try:
            data = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.load_error = (
                f"Could not read the existing ratings/status at {state_path} ({exc}). "
                f"Starting unrated -- avoid marking/rating photos until this is resolved, "
                f"since doing so would save over (and permanently lose) the unreadable file."
            )
            return
        if not isinstance(data, dict):
            self.load_error = (
                f"The existing ratings/status file at {state_path} isn't in the expected "
                f"format. Starting unrated -- avoid marking/rating photos until this is "
                f"resolved, since doing so would save over (and permanently lose) the file."
            )
            return

        renamed_names = data.get(RENAMED_NAMES_KEY)
        if isinstance(renamed_names, dict):
            self.renamed_names = {
                str(name): number for name, number in renamed_names.items() if isinstance(number, int)
            }

        by_name = {item.name: item for item in self.items}
        for name, entry in data.items():
            if name == RENAMED_NAMES_KEY:
                continue
            item = by_name.get(name)
            if item is None:
                continue
            status_value = entry.get("status")
            if status_value in (s.value for s in Status):
                item.status = Status(status_value)
            rating = entry.get("rating")
            if isinstance(rating, int) and 0 <= rating <= 5:
                item.rating = rating

    def save_state(self) -> None:
        if self.folder is None:
            return
        data = {
            item.name: {"status": item.status.value, "rating": item.rating}
            for item in self.items
            if item.status is not Status.UNRATED or item.rating != 0
        }
        if self.renamed_names:
            data[RENAMED_NAMES_KEY] = self.renamed_names
        atomic_write_bytes(self._state_path(), json.dumps(data, indent=2).encode("utf-8"))

    def register_name_use(self, name: str) -> int:
        """Record a use of `name`, returning the next unused sequence number for it."""
        number = self.renamed_names.get(name, 0) + 1
        self.renamed_names[name] = number
        return number

    def sort_items(self, key: Callable[[ImageItem], Any]) -> None:
        """Re-sort `items` by `key`, keeping `current_index` pointed at the same item."""
        current = self.current_item
        self.items.sort(key=key)
        if current is not None:
            self.current_index = self.items.index(current)

    @property
    def current_item(self) -> ImageItem | None:
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
        return None

    def next(self) -> None:
        if self.current_index < len(self.items) - 1:
            self.current_index += 1

    def prev(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1

    def set_status(self, index: int, status: Status) -> None:
        if 0 <= index < len(self.items):
            self.items[index].status = status

    def set_rating(self, index: int, rating: int) -> None:
        if 0 <= index < len(self.items) and 0 <= rating <= 5:
            self.items[index].rating = rating

    def counts(self) -> dict[str, int]:
        result = {"selected": 0, "rejected": 0, "unrated": 0}
        for item in self.items:
            result[item.status.value] += 1
        return result
