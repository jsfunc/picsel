"""Per-folder cache of detected faces, persisted alongside the photos.

Detection+embedding is run once per image and cached to `.tamis_faces.json`
in the photo's own folder (mirroring `.tamis_state.json`'s per-folder
sidecar convention) -- a confidence-threshold slider in the UI then just
re-filters the cached list, and never needs to re-run detection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tamis.persistence import atomic_write_bytes
from tamis.recognition.codec import decode_embedding, encode_embedding
from tamis.recognition.detector import DEFAULT_MIN_CONFIDENCE, FaceDetection, detect_faces, load_for_detection
from tamis.recognition.embedder import embed_faces

logger = logging.getLogger(__name__)

FACES_FILENAME = ".tamis_faces.json"
_LEGACY_FACES_FILENAME = ".picsel_faces.json"  # pre-rename name (picSel -> Tamis); migrated automatically


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
        # Set by load() only when this folder's sidecar file *exists* but
        # couldn't be read -- as opposed to legitimately not existing yet
        # (a folder with no faces detected yet). Mirrors PersonGallery's
        # load_error: save() has no "don't overwrite if empty" guard beyond
        # the blanket `if not self._records: return` below, which would
        # *not* protect a corrupted file once even one face gets detected
        # fresh in this session. A caller should check this after every
        # load() and warn before anything can save() over the unreadable
        # file.
        self.load_error: str | None = None

    def _migrate_legacy_faces_file(self) -> None:
        """One-time migration for a folder opened before the picSel -> Tamis
        rename: if the old sidecar exists and the new one doesn't yet, rename
        it in place. Best-effort -- a failure here just means load() starts
        fresh, same as if the folder had never been opened before."""
        assert self.folder is not None
        new_path = self._state_path()
        legacy_path = self.folder / _LEGACY_FACES_FILENAME
        if new_path.exists() or not legacy_path.exists():
            return
        try:
            legacy_path.rename(new_path)
        except OSError as exc:
            logger.warning("Could not migrate %s to %s: %s", legacy_path, new_path, exc)

    def load(self, folder: Path) -> None:
        self.folder = Path(folder)
        self._records = {}
        self._generation += 1
        self.load_error = None
        self._migrate_legacy_faces_file()
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.load_error = (
                f"Could not read the existing face data at {path} ({exc}). "
                f"Starting empty for this folder -- avoid confirming any "
                f"face names until this is resolved, since doing so would "
                f"save over (and permanently lose) the unreadable file."
            )
            return
        for name, entries in data.items():
            self._records[name] = [
                FaceRecord(
                    box=tuple(entry["box"]),
                    confidence=entry.get("confidence"),
                    embedding=decode_embedding(entry["embedding"]),
                    dismissed=entry.get("dismissed", False),
                    person_id=entry.get("person_id"),
                )
                for entry in entries
            ]

    def save(self) -> None:
        prepared = self.prepare_save()
        if prepared is None:
            return
        path, data = prepared
        self.write_payload(path, data)

    def prepare_save(self) -> tuple[Path, dict] | None:
        """Synchronously snapshot the current records into whatever save()
        would write, without doing the actual (slow) JSON serialization or
        disk write -- the split point that lets a caller (see
        FaceRecognitionController) capture a consistent, fully up-to-date
        snapshot on the calling thread, then defer the expensive part to a
        background thread. Safe against a later folder switch, since
        `load()` *reassigns* self._records/self.folder rather than mutating
        them in place -- this snapshot doesn't see that reassignment."""
        if self.folder is None:
            return None
        if not self._records:
            return None
        data = {
            name: [
                {
                    "box": list(record.box),
                    "confidence": record.confidence,
                    "embedding": encode_embedding(record.embedding),
                    "dismissed": record.dismissed,
                    "person_id": record.person_id,
                }
                for record in records
            ]
            # list(...) snapshots the dict before iterating -- see
            # remap_person's comment: faces_for() can insert a new key from
            # a background worker thread while this runs on the GUI thread.
            # save() is the most frequently called of the four methods that
            # touch self._records, so it's the one most likely to actually
            # race in practice.
            for name, records in list(self._records.items())
        }
        return self._state_path(), data

    @staticmethod
    def write_payload(path: Path, data: dict) -> None:
        atomic_write_bytes(path, json.dumps(data, indent=2).encode("utf-8"))

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

    def reconcile_people(self, redirects: dict[str, str], known_ids: set[str]) -> int:
        """Bring this folder's labels back in line with the gallery, following
        merge redirects and clearing labels naming someone who no longer
        exists. Returns how many records changed.

        Called on load, which is what makes a merge or a "forget" apply to
        *every* folder rather than only whichever one happened to be open when
        it was done. Previously only the open folder was repaired (see
        `remap_person`), so records elsewhere kept pointing at a deleted id --
        and since the labeling path reads an unresolvable id as "not labeled
        yet", re-confirming such a face added a second copy of its sample to
        the gallery instead of moving the existing one.
        """
        changed = 0
        for records in list(self._records.values()):  # see remap_person for why list(...)
            for record in records:
                if record.person_id is None:
                    continue
                resolved: str | None = record.person_id
                seen: set[str] = set()
                while resolved in redirects and resolved not in seen:
                    seen.add(resolved)
                    resolved = redirects[resolved]
                if resolved not in known_ids:
                    resolved = None  # merged away with no successor, or forgotten entirely
                if resolved != record.person_id:
                    record.person_id = resolved
                    changed += 1
        if changed:
            logger.info("Reconciled %d face label(s) against the gallery in %s", changed, self.folder)
        return changed

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
