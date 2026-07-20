"""File system operations: unique naming and applying culling decisions."""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image

from picsel.models.image_item import ImageItem, Status
from picsel.models.library import ImageLibrary


def unique_path(path: Path) -> Path:
    """Return `path`, or `path` with a ` (1)`, ` (2)`, ... suffix if it already exists."""
    if not path.exists():
        return path
    parent, stem, suffix = path.parent, path.stem, path.suffix
    counter = 1
    while True:
        candidate = parent / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def rename_with_sequence(item: ImageItem, name: str, number: int) -> Path:
    """Rename `item`'s file to `<name><number, zero-padded to 3 digits>`, keeping its extension.

    Updates `item.path` in place and returns the new path.
    """
    new_name = f"{name}{number:03d}{item.path.suffix}"
    destination = unique_path(item.path.parent / new_name)
    item.path.rename(destination)
    item.path = destination
    return destination


SEQUENCE_NAME_RE = re.compile(r"^(.*?)(\d+)$")

_EXIF_SUBIFD_TAG = 0x8769  # "Exif IFD Pointer": where cameras actually store DateTimeOriginal
_DATETIME_ORIGINAL_TAG = 0x9003
_DATETIME_TAG = 0x0132  # IFD0 fallback, set by some tools instead of DateTimeOriginal
_EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


def capture_time(path: Path) -> float:
    """Best-effort timestamp for when a photo was taken.

    Prefers EXIF DateTimeOriginal (in the Exif sub-IFD, where cameras put it —
    not IFD0), falls back to DateTime, then to the file's mtime if there is no
    usable EXIF data at all.
    """
    raw = None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            raw = exif.get_ifd(_EXIF_SUBIFD_TAG).get(_DATETIME_ORIGINAL_TAG) or exif.get(_DATETIME_TAG)
    except Exception:
        raw = None

    if raw:
        try:
            return datetime.strptime(raw, _EXIF_DATETIME_FORMAT).timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def find_sequence_groups(items: list[ImageItem]) -> dict[str, list[ImageItem]]:
    """Group items named `<name><digits><ext>` by `<name>`.

    Only groups with 2+ members are returned, since a lone file has nothing to
    reorder relative to.
    """
    groups: dict[str, list[ImageItem]] = {}
    for item in items:
        match = SEQUENCE_NAME_RE.match(item.path.stem)
        if not match or not match.group(1):
            continue
        groups.setdefault(match.group(1), []).append(item)
    return {name: group for name, group in groups.items() if len(group) > 1}


@dataclass
class RenumberReport:
    renamed: int = 0
    errors: list[str] = field(default_factory=list)


def renumber_by_creation_time(items: list[ImageItem], name: str) -> RenumberReport:
    """Renumber all `<name><digits><ext>` items so their sequence number matches
    capture-time order (earliest first), e.g. if `toto001.jpg` was actually taken
    after `toto002.jpg`, they get swapped.

    Updates each renamed item's `.path` in place. Renames go through unique
    temporary names first, so arbitrary reorderings (not just simple swaps)
    can't collide with each other mid-rename.
    """
    matched = [
        item
        for item in items
        if (match := SEQUENCE_NAME_RE.match(item.path.stem)) and match.group(1) == name
    ]
    report = RenumberReport()
    if len(matched) < 2:
        return report

    ordered = sorted(matched, key=lambda item: capture_time(item.path))
    width = max(3, len(str(len(ordered))))

    staged: list[tuple[ImageItem, Path]] = []
    for item in ordered:
        temp = unique_path(item.path.with_name(f".picsel_renumber_{uuid.uuid4().hex}{item.path.suffix}"))
        try:
            item.path.rename(temp)
            staged.append((item, temp))
        except OSError as exc:
            report.errors.append(f"{item.path.name}: {exc}")

    for index, (item, temp) in enumerate(staged, start=1):
        target = unique_path(temp.with_name(f"{name}{index:0{width}d}{temp.suffix}"))
        try:
            temp.rename(target)
            item.path = target
            report.renamed += 1
        except OSError as exc:
            # The file is sitting at `temp` on disk now, not its original path;
            # keep the item's bookkeeping pointed at where it actually is.
            item.path = temp
            report.errors.append(f"{temp.name}: {exc}")

    return report


_DATE_RENAME_FORMAT = "p%Y%m%d_%H%M%S"


def rename_by_creation_date(items: list[ImageItem]) -> RenumberReport:
    """Rename every item to `pYYYYmmdd_hhmmss<ext>` based on its capture time
    (see `capture_time`), e.g. `p20240315_143022.jpg`.

    Updates each renamed item's `.path` in place. Renames go through unique
    temporary names first, so two items landing on the same target name (e.g.
    a burst shot within the same second) can't collide mid-rename; the
    later one gets a ` (1)`, ` (2)`, ... suffix via `unique_path`.
    """
    report = RenumberReport()
    if not items:
        return report

    targets = [
        (item, datetime.fromtimestamp(capture_time(item.path)).strftime(_DATE_RENAME_FORMAT) + item.path.suffix)
        for item in items
    ]

    staged: list[tuple[ImageItem, Path, str]] = []
    for item, name in targets:
        temp = unique_path(item.path.with_name(f".picsel_rename_{uuid.uuid4().hex}{item.path.suffix}"))
        try:
            item.path.rename(temp)
            staged.append((item, temp, name))
        except OSError as exc:
            report.errors.append(f"{item.path.name}: {exc}")

    for item, temp, name in staged:
        target = unique_path(temp.with_name(name))
        try:
            temp.rename(target)
            item.path = target
            report.renamed += 1
        except OSError as exc:
            # The file is sitting at `temp` on disk now, not its original path;
            # keep the item's bookkeeping pointed at where it actually is.
            item.path = temp
            report.errors.append(f"{temp.name}: {exc}")

    return report


@dataclass
class CullingReport:
    moved_selected: int = 0
    moved_rejected: int = 0
    errors: list[str] = field(default_factory=list)


def apply_culling(
    library: ImageLibrary,
    mode: str = "move",
    selected_dir: str = "selected",
    rejected_dir: str = "rejected",
) -> CullingReport:
    """Move or copy SELECTED/REJECTED items into subfolders of the library folder.

    UNRATED items are left in place. For `mode="move"`, each moved item's
    `path` is updated to its new location. Returns a report with counts and
    any per-file errors.
    """
    if library.folder is None:
        raise ValueError("Library has no folder loaded")
    if mode not in ("move", "copy"):
        raise ValueError(f"Invalid mode: {mode!r}")

    report = CullingReport()
    target_dirs = {
        Status.SELECTED: library.folder / selected_dir,
        Status.REJECTED: library.folder / rejected_dir,
    }

    for item in library.items:
        target_root = target_dirs.get(item.status)
        if target_root is None:
            continue
        try:
            target_root.mkdir(parents=True, exist_ok=True)
            destination = unique_path(target_root / item.path.name)
            if mode == "move":
                shutil.move(str(item.path), str(destination))
                item.path = destination
            else:
                shutil.copy2(str(item.path), str(destination))
            if item.status is Status.SELECTED:
                report.moved_selected += 1
            else:
                report.moved_rejected += 1
        except OSError as exc:
            report.errors.append(f"{item.path.name}: {exc}")

    return report
