"""Per-folder cache of detected faces, persisted alongside the photos.

Detection+embedding is run once per image and cached to `.picsel_faces.json`
in the photo's own folder (mirroring `.picsel_state.json`'s per-folder
sidecar convention) -- a confidence-threshold slider in the UI then just
re-filters the cached list, and never needs to re-run detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from picsel.recognition.detector import DEFAULT_MIN_CONFIDENCE, FaceDetection, detect_faces, load_for_detection
from picsel.recognition.embedder import embed_faces

FACES_FILENAME = ".picsel_faces.json"


@dataclass(eq=False)
class FaceRecord:
    """`eq=False` is deliberate: every caller that compares FaceRecords
    (`in`, `.remove()`, `.index()`) wants identity, not value, equality --
    e.g. `remove_manual_face` finds a specific record among others that may
    share the same box/confidence. The default dataclass `__eq__` would
    compare `.embedding` too (a numpy array), and comparing two *different*
    array objects with `==` raises "truth value of an array is ambiguous"
    the moment an earlier field (box, confidence) happens to also match --
    e.g. two manually-added boxes accidentally drawn at the same spot.
    """

    box: tuple[int, int, int, int]
    confidence: float | None  # None marks a manually-added face ("add a face here")
    embedding: np.ndarray
    dismissed: bool = False  # True marks a detector box the user rejected ("no face here")
    person_id: str | None = None

    @property
    def is_manual(self) -> bool:
        return self.confidence is None


class FaceCatalog:
    """Caches face detection/embedding results per image within one folder,
    and tracks manual overrides and person labels on top of them."""

    def __init__(self) -> None:
        self.folder: Path | None = None
        self._records: dict[str, list[FaceRecord]] = {}
        # Bumped on every load() (folder switch). faces_for() captures the
        # generation before running slow detect+embed work on a background
        # thread and only writes the result into self._records if the
        # catalog hasn't since moved on to a different folder -- otherwise a
        # worker still running for the *previous* folder could write into
        # the new folder's _records dict (keyed by filename only, so a
        # same-named file in both folders -- common with camera default
        # naming -- would get a completely wrong cached entry).
        self._generation = 0

    def load(self, folder: Path) -> None:
        self.folder = Path(folder)
        self._records = {}
        self._generation += 1
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for name, entries in data.items():
            self._records[name] = [
                FaceRecord(
                    box=tuple(entry["box"]),
                    confidence=entry.get("confidence"),
                    embedding=np.array(entry["embedding"], dtype=np.float32),
                    dismissed=entry.get("dismissed", False),
                    person_id=entry.get("person_id"),
                )
                for entry in entries
            ]

    def save(self) -> None:
        if self.folder is None:
            return
        if not self._records:
            return
        data = {
            name: [
                {
                    "box": list(record.box),
                    "confidence": record.confidence,
                    "embedding": record.embedding.tolist(),
                    "dismissed": record.dismissed,
                    "person_id": record.person_id,
                }
                for record in records
            ]
            for name, records in self._records.items()
        }
        self._state_path().write_text(json.dumps(data, indent=2))

    def _state_path(self) -> Path:
        assert self.folder is not None
        return self.folder / FACES_FILENAME

    def faces_for(self, path: Path) -> list[FaceRecord]:
        """Return every cached face record for `path` (dismissed and below-threshold
        included), running detection+embedding the first time this image is seen."""
        name = path.name
        if name not in self._records:
            generation = self._generation
            image = load_for_detection(path)
            detections = detect_faces(image)
            embeddings = embed_faces(image, detections)
            records = [
                FaceRecord(box=d.box, confidence=d.confidence, embedding=e)
                for d, e in zip(detections, embeddings)
            ]
            if generation != self._generation:
                # load() switched folders while detection was running (this
                # can be a slow background-thread call) -- don't let a stale
                # result from the old folder pollute the new one's cache.
                # Return it to this caller anyway (still a correct answer for
                # the path actually asked about), just don't cache it.
                return records
            self._records[name] = records
        return self._records[name]

    def invalidate(self, path: Path) -> None:
        """Drop `path`'s cached detection/embedding/label data, if any -- used
        after overwriting the file with an edited (rotated/flipped/cropped)
        version, since the old cached boxes/embeddings were computed against
        the previous pixel geometry and would otherwise be silently reused as
        if still correct. The next `faces_for` call re-detects from scratch.
        """
        self._records.pop(path.name, None)

    def visible_faces(self, path: Path, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> list[FaceRecord]:
        """`faces_for`, filtered for display: dismissed records dropped, and
        detector-found records below `min_confidence` dropped. Manually-added
        records have no confidence score to filter by, so they're always shown.
        """
        return [
            record
            for record in self.faces_for(path)
            if not record.dismissed and (record.is_manual or record.confidence >= min_confidence)
        ]

    def add_manual_face(self, path: Path, box: tuple[int, int, int, int]) -> FaceRecord:
        """Add a user-drawn face box ("add a face here") and embed it immediately."""
        image = load_for_detection(path)
        placeholder = FaceDetection(box=box, confidence=1.0)  # confidence unused; embed_faces only reads .box
        [embedding] = embed_faces(image, [placeholder])
        record = FaceRecord(box=box, confidence=None, embedding=embedding)
        self.faces_for(path).append(record)
        return record

    def remove_manual_face(self, path: Path, record: FaceRecord) -> None:
        records = self.faces_for(path)
        if record.is_manual and record in records:
            records.remove(record)

    def dismiss(self, record: FaceRecord) -> None:
        """Mark a detector-found box as not a face ("no face here")."""
        record.dismissed = True

    def assign_person(self, record: FaceRecord, person_id: str | None) -> None:
        record.person_id = person_id

    def remap_person(self, old_person_id: str, new_person_id: str) -> None:
        """Point every cached record assigned to `old_person_id` at
        `new_person_id` instead -- used after two PersonGallery entries get
        merged into one, so labels in this folder don't go dangling."""
        # list(...) snapshots the dict's values before iterating: faces_for()
        # can insert a new key into self._records from a background
        # FaceDetectionWorker/FolderSearchWorker thread while this runs on
        # the GUI thread, and iterating the live dict during that would
        # raise "dictionary changed size during iteration".
        for records in list(self._records.values()):
            for record in records:
                if record.person_id == old_person_id:
                    record.person_id = new_person_id

    def forget_person(self, person_id: str) -> None:
        """Clear `person_id`'s label from every cached record in this folder
        -- used when a person is removed entirely from PersonGallery, so
        labels in this folder don't dangle."""
        for records in list(self._records.values()):  # see remap_person for why list(...)
            for record in records:
                if record.person_id == person_id:
                    record.person_id = None

    def unassign_all_people(self) -> None:
        """Clear every cached record's person label -- used by "forget all
        faces" so labels in this folder don't dangle after PersonGallery is
        wiped."""
        for records in list(self._records.values()):  # see remap_person for why list(...)
            for record in records:
                record.person_id = None
