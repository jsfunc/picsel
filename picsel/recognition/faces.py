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


@dataclass
class FaceRecord:
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

    def load(self, folder: Path) -> None:
        self.folder = Path(folder)
        self._records = {}
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
            image = load_for_detection(path)
            detections = detect_faces(image)
            embeddings = embed_faces(image, detections)
            self._records[name] = [
                FaceRecord(box=d.box, confidence=d.confidence, embedding=e)
                for d, e in zip(detections, embeddings)
            ]
        return self._records[name]

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
        for records in self._records.values():
            for record in records:
                if record.person_id == old_person_id:
                    record.person_id = new_person_id

    def unassign_all_people(self) -> None:
        """Clear every cached record's person label -- used by "forget all
        faces" so labels in this folder don't dangle after PersonGallery is
        wiped."""
        for records in self._records.values():
            for record in records:
                record.person_id = None
