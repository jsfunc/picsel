"""Central, cross-folder store of known people and their embedding prototypes.

A person is identified by a growable set of embedding samples rather than a
fixed classifier output, so adding a new person -- or a new sample for an
existing one -- is just appending to a list, never retraining anything.
"""

from __future__ import annotations

import gzip
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tamis.persistence import atomic_write_bytes
from tamis.recognition.codec import decode_embedding, duplicate_key, encode_embedding

logger = logging.getLogger(__name__)

DEFAULT_GALLERY_PATH = Path.home() / ".tamis" / "people.json.gz"
# Pre-rename location (picSel -> Tamis) -- only ever consulted as a fallback
# when using DEFAULT_GALLERY_PATH itself, never for a caller-supplied custom
# path (e.g. in tests), so this can never reach outside the real gallery's
# own home-directory location.
_LEGACY_GALLERY_PATH = Path.home() / ".picsel" / "people.json.gz"

# How many of the single nearest embedding *samples* (across all people, not
# per person) to consider when picking the single best-guess suggestion.
# Looking only at the single nearest sample (k=1) means one lucky/unlucky
# sample can dominate the result; pooling a neighborhood of samples is
# sturdier. There's no hard "recognized or not" threshold on top of this --
# see `identify`'s docstring -- so a UI showing its result should convey
# confidence continuously (e.g. by color) rather than as a yes/no gate.
DEFAULT_K = 10


@dataclass
class Person:
    id: str
    name: str
    embeddings: list[np.ndarray] = field(default_factory=list)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class PersonGallery:
    """Persistent registry of known people, loaded from and saved to a single
    JSON file shared across the whole photo library (not per-folder)."""

    def __init__(self, path: Path = DEFAULT_GALLERY_PATH) -> None:
        self.path = Path(path)
        self.people: list[Person] = []
        # Set only when a gallery file *exists* but couldn't be read -- as
        # opposed to legitimately not existing yet, which is a normal, silent
        # empty start. Distinguishing the two matters because save() has no
        # "don't overwrite if empty" guard (unlike FaceCatalog.save()) --
        # Forget All Faces legitimately empties the gallery and must still be
        # able to persist that. So a caller should check this right after
        # construction and warn the user before anything else runs a save(),
        # since that save would otherwise permanently overwrite the
        # unreadable-but-still-present file with an empty one.
        self.load_error: str | None = None
        # Redirects left behind by merge(): {merged-away id: surviving id}.
        # Persisted, because a face record naming the merged-away person can
        # live in *any* folder's sidecar, and only the folder open at merge
        # time used to get repaired -- see `merge`'s docstring.
        self.merged_ids: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        # Embeddings are 512 floats each and accumulate one per confirmed
        # face, so this file grows steadily -- gzip shrinks the (very
        # repetitive) JSON text substantially.
        if self.path.exists():
            source, compressed = self.path, True
        else:
            source = compressed = None
            # Fallback chain, most to least recent: a pre-compression
            # plain-JSON gallery at the same location minus ".gz" (so
            # existing data keeps loading after that format change), then --
            # only for the real default path, never a caller-supplied custom
            # one -- the pre-rename (picSel -> Tamis) ~/.picsel/ location,
            # compressed and its own plain-JSON fallback. None of these are
            # deleted after migrating; save() always (re)writes the current
            # compressed DEFAULT_GALLERY_PATH.
            candidates = []
            if self.path.suffix == ".gz":
                candidates.append((self.path.with_suffix(""), False))
            if self.path == DEFAULT_GALLERY_PATH:
                candidates.append((_LEGACY_GALLERY_PATH, True))
                candidates.append((_LEGACY_GALLERY_PATH.with_suffix(""), False))
            for candidate_path, candidate_compressed in candidates:
                if candidate_path.exists():
                    source, compressed = candidate_path, candidate_compressed
                    break
            if source is None:
                return
        try:
            raw = source.read_bytes()
            if compressed:
                raw = gzip.decompress(raw)
            data = json.loads(raw)
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
            self.load_error = (
                f"Could not read the existing face gallery at {source} ({exc}). "
                f"Starting empty -- avoid confirming any face names until this is "
                f"resolved, since doing so would save over (and permanently lose) "
                f"the unreadable file."
            )
            return
        self.people = [
            Person(
                id=entry["id"],
                name=entry["name"],
                embeddings=[decode_embedding(e) for e in entry.get("embeddings", [])],
            )
            for entry in data.get("people", [])
        ]
        stored_redirects = data.get("merged_ids")
        if isinstance(stored_redirects, dict):
            self.merged_ids = {
                str(old): str(new) for old, new in stored_redirects.items() if isinstance(new, str)
            }
        self._drop_duplicate_samples()

    def _drop_duplicate_samples(self) -> int:
        """Remove samples that are byte-identical to one already held, and
        samples filed under two people at once. Returns how many were dropped.

        Run on load rather than offered as a manual repair, because until the
        duplicate guard in `add_embedding` existed, several ordinary actions
        quietly produced these -- a `merge` concatenated both entries' samples
        with no deduplication, and confirming a name on a face whose stored
        person id had been merged away took the "this face has no person yet"
        branch, adding a second copy instead of moving the existing one. A
        real gallery built before those fixes was measured at 242 samples of
        which 83 (34%) were redundant.

        A sample held by *two different people* is worse than redundant: the
        same face votes for both, so `identify` returns an exact tie and picks
        between them arbitrarily (observed on real data, at similarity
        1.000000). There is nothing recorded that says which label came last,
        so there is no principled way to keep one -- it is dropped from both,
        leaving each person their other samples and removing a guess that was
        never better than a coin flip.

        Keyed on the encoded form (see codec.duplicate_key) so this is a
        linear dict pass, not an O(n^2) sweep of array comparisons.
        """
        owners: dict[str, set[str]] = {}
        for person in self.people:
            for embedding in person.embeddings:
                owners.setdefault(duplicate_key(embedding), set()).add(person.id)
        contested = {key for key, ids in owners.items() if len(ids) > 1}

        dropped = 0
        for person in self.people:
            kept: list[np.ndarray] = []
            seen: set[str] = set()
            for embedding in person.embeddings:
                key = duplicate_key(embedding)
                if key in contested or key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                kept.append(embedding)
            person.embeddings = kept
        if dropped:
            logger.info(
                "Dropped %d duplicate/contested embedding sample(s) while loading %s", dropped, self.path
            )
        return dropped

    def save(self) -> None:
        path, data = self.prepare_save()
        self.write_payload(path, data)

    def prepare_save(self) -> tuple[Path, dict]:
        """Synchronously snapshot the current people/embeddings into whatever
        save() would write, without doing the actual (slow) gzip+JSON
        serialization or disk write -- the split point that lets a caller
        (see FaceRecognitionController) capture a consistent, fully
        up-to-date snapshot on the calling thread, then defer the expensive
        part to a background thread."""
        return self.path, self._snapshot()

    def _snapshot(self) -> dict:
        snapshot: dict = {
            "people": [
                {"id": p.id, "name": p.name, "embeddings": [encode_embedding(e) for e in p.embeddings]}
                for p in self.people
            ]
        }
        if self.merged_ids:
            snapshot["merged_ids"] = dict(self.merged_ids)
        return snapshot

    def export_to(self, path: Path) -> None:
        """Write this whole gallery, gzip-compressed, to an arbitrary path --
        `save()`'s own mechanism, also exposed directly for the user to
        export a portable copy (e.g. to back up or move to another machine).
        """
        self.write_payload(Path(path), self._snapshot())

    @staticmethod
    def write_payload(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, gzip.compress(json.dumps(data).encode("utf-8")))

    def import_from(self, path: Path) -> int:
        """Merge people from a gallery file exported by `export_to` (or this
        gallery's own save file) into this one. Accepts either gzip-
        compressed or plain JSON, so a manually-edited or older export still
        works. A person whose name exactly matches an existing one has their
        embedding samples appended to that existing person; everyone else is
        added as a new person with a freshly generated id, never reusing the
        imported file's id, to avoid collisions between independent galleries.
        Returns how many *new* people were added (not merged into existing).
        """
        raw = Path(path).read_bytes()
        try:
            raw = gzip.decompress(raw)
        except gzip.BadGzipFile:
            pass  # a plain (uncompressed) export -- read as-is
        data = json.loads(raw)

        # Parse every entry before mutating self.people at all: a malformed
        # entry partway through the file (e.g. missing "name") must not
        # leave earlier entries already merged while the caller is told the
        # whole import failed.
        parsed = [
            (entry["name"], [decode_embedding(e) for e in entry.get("embeddings", [])])
            for entry in data.get("people", [])
        ]

        added = 0
        for name, embeddings in parsed:
            existing = self.find_by_name(name)
            if existing is not None:
                # Via add_embedding, not extend: re-importing a gallery that
                # overlaps this one (a backup, or the same export twice) would
                # otherwise duplicate every shared sample.
                for embedding in embeddings:
                    self.add_embedding(existing.id, embedding)
            else:
                person = self.add_person(name)
                for embedding in embeddings:
                    self.add_embedding(person.id, embedding)
                added += 1
        return added

    def find_by_id(self, person_id: str) -> Person | None:
        """Look up a person, following any redirect left by a `merge`.

        Resolving here (rather than only rewriting the face records of
        whichever folder happened to be open when the merge ran) is what makes
        a merge take effect everywhere: a sidecar in a folder that was closed
        at the time still names the merged-away person, and without this it
        would read as "unknown person" -- which the labeling path used to
        treat as "not labeled yet", quietly adding a duplicate sample instead
        of moving the existing one.
        """
        return next((p for p in self.people if p.id == self.resolve_id(person_id)), None)

    def resolve_id(self, person_id: str) -> str:
        """Follow merge redirects to the id that actually holds this person's
        samples now. Chains are followed (a was merged into b, b later into c),
        with a bound so a corrupted file containing a cycle can't hang."""
        seen: set[str] = set()
        while person_id in self.merged_ids and person_id not in seen:
            seen.add(person_id)
            person_id = self.merged_ids[person_id]
        return person_id

    def find_by_name(self, name: str) -> Person | None:
        return next((p for p in self.people if p.name == name), None)

    def similarity_to(self, person_id: str, embedding: np.ndarray) -> float:
        """Cosine similarity between `embedding` and `person_id`'s closest
        sample, or 0.0 if the person is unknown or has no samples yet."""
        person = self.find_by_id(person_id)
        if person is None or not person.embeddings:
            return 0.0
        return max(_cosine_similarity(embedding, sample) for sample in person.embeddings)

    def add_person(self, name: str) -> Person:
        person = Person(id=uuid.uuid4().hex, name=name)
        self.people.append(person)
        return person

    def remove_person(self, person_id: str) -> None:
        self.people = [p for p in self.people if p.id != person_id]

    def add_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        """Add a reference sample for `person_id`, ignoring a sample this
        person already holds.

        The duplicate guard makes this idempotent, which matters because a
        sample carries no identity of its own: nothing links it back to the
        face record it came from, so a caller that adds the same face twice
        has no way to notice or undo it. Re-adding used to be reachable from
        several ordinary paths (see `_drop_duplicate_samples`) and inflated a
        real gallery by 34%, slowing every save without improving matching --
        duplicate samples don't strengthen a match, since `similarity_to` and
        `rank_all` both score a person by their single closest sample.
        """
        person = self.find_by_id(person_id)
        if person is None:
            raise ValueError(f"Unknown person id: {person_id!r}")
        embedding = np.asarray(embedding, dtype=np.float32)
        key = duplicate_key(embedding)
        if any(duplicate_key(sample) == key for sample in person.embeddings):
            return
        person.embeddings.append(embedding)

    def remove_embedding(self, person_id: str, embedding: np.ndarray) -> bool:
        """Remove `embedding` from `person_id`'s samples, matched by value
        (not identity -- a reloaded-from-disk array is never the same object).

        Used when a face gets relabeled or unassigned: the sample added under
        the old label no longer describes that face, so it shouldn't keep
        sitting in that person's reference data. Returns whether a matching
        sample was found and removed.

        If this empties `person_id`'s samples entirely, the person is removed
        from the gallery too -- there's no dedicated "delete this person"
        action in the UI, so leaving a permanent zero-sample stub behind
        (invisible to matching, but cluttering the Manage People list) would
        otherwise be the only way this situation could arise.
        """
        person = self.find_by_id(person_id)
        if person is None:
            return False
        key = duplicate_key(embedding)
        for i, sample in enumerate(person.embeddings):
            # Matched on the encoded form rather than a float tolerance: one
            # side of this comparison is typically a freshly-computed float32
            # and the other came back through the sidecar. See
            # codec.duplicate_key for why that is exact.
            if duplicate_key(sample) == key:
                del person.embeddings[i]
                if not person.embeddings:
                    # person.id, not person_id: the argument may be an id that
                    # was merged away, which find_by_id resolved to whoever
                    # holds the samples now -- removing by the unresolved id
                    # would match nobody and strand a zero-sample person.
                    self.remove_person(person.id)
                return True
        return False

    def merge(self, keep_id: str, remove_id: str) -> None:
        """Fold `remove_id`'s embedding samples into `keep_id` and delete
        `remove_id` -- e.g. to fix a duplicate created by entering the same
        person's name with different capitalization ("papa" vs "Papa").
        `keep_id`'s name is left as-is; callers wanting a different combined
        name should set `person.name` themselves after merging.

        Samples already held by `keep_id` are not copied over again: the same
        face is routinely labeled in more than one folder (the face cache is
        per-folder, this gallery is global), so the two entries being merged
        very often overlap.

        Records the merge in `merged_ids` so face records naming `remove_id`
        keep resolving -- see `find_by_id`. Only the folder open at merge time
        can have its sidecar rewritten, and every other folder's records would
        otherwise be left pointing at an id that no longer exists.
        """
        if keep_id == remove_id:
            raise ValueError("Cannot merge a person into themself")
        keep = self.find_by_id(keep_id)
        remove = self.find_by_id(remove_id)
        if keep is None or remove is None:
            raise ValueError("Unknown person id")
        if keep.id == remove.id:
            raise ValueError("Cannot merge a person into themself")  # both ids resolved to one person
        for sample in remove.embeddings:
            self.add_embedding(keep.id, sample)
        # remove.id / keep.id rather than the arguments: either may be an id
        # that a previous merge already redirected, and remove_person matches
        # on the real id only.
        self.remove_person(remove.id)
        self.merged_ids[remove.id] = keep.id
        self.merged_ids[remove_id] = keep.id
        # Re-point anything that already redirected to the person just merged
        # away, so lookups stay one hop and can't depend on chain length.
        # list(...) because this assigns into the dict it is walking.
        for old_id, new_id in list(self.merged_ids.items()):
            if new_id == remove.id:
                self.merged_ids[old_id] = keep.id

    def identify(self, embedding: np.ndarray, k: int = DEFAULT_K) -> list[tuple[Person, float]]:
        """Return candidate people for `embedding`, ranked most-similar first.

        Finds the `k` nearest embedding *samples* to `embedding` across every
        person's samples pooled together (not just each person's single
        closest sample), then collapses those down to the distinct people
        represented, each scored by its closest sample among that top-k set.
        This makes the result a short, ranked shortlist rather than one
        all-or-nothing 1-nearest-neighbor guess -- e.g. if a person's most
        similar sample and a different person's next-most-similar sample are
        both close, both show up, correctly ordered.

        No similarity floor is applied -- there's no single threshold that
        cleanly separates "recognized" from "not" (see the recognition
        feature's design notes: a genuine cross-photo match scored as low as
        0.555, while unrelated siblings in the same photo scored 0.61 against
        each other), so callers should convey confidence continuously (e.g.
        coloring a suggestion by its similarity) rather than filtering by a
        cutoff. A person appears at most once, even if several of their
        samples land in the top `k`.
        """
        samples = [
            (_cosine_similarity(embedding, sample), person)
            for person in self.people
            for sample in person.embeddings
        ]
        samples.sort(key=lambda item: item[0], reverse=True)

        ranked: list[tuple[Person, float]] = []
        seen_ids: set[str] = set()
        for similarity, person in samples[:k]:
            if person.id in seen_ids:
                continue
            seen_ids.add(person.id)
            ranked.append((person, similarity))

        return ranked

    def rank_all(self, embedding: np.ndarray) -> list[tuple[Person, float]]:
        """Return every person who has at least one embedding sample, each
        scored by their closest sample to `embedding`, sorted most-similar
        first -- no `k` cutoff and no `min_similarity` floor.

        Unlike `identify` (a confident, threshold-gated auto-suggestion),
        this is meant for a manual pick-list -- e.g. a dropdown -- where
        showing everyone, worst matches included, lets the user correct a
        mismatch or find someone the confident-match threshold excluded.
        """
        ranked = [
            (person, max(_cosine_similarity(embedding, sample) for sample in person.embeddings))
            for person in self.people
            if person.embeddings
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
