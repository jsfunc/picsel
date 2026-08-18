"""Search a folder's photos for occurrences of a specific known person."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from picsel.recognition.detector import DEFAULT_MIN_CONFIDENCE
from picsel.recognition.faces import FaceCatalog, FaceRecord
from picsel.recognition.gallery import Person, PersonGallery


@dataclass
class SearchHit:
    path: Path
    record: FaceRecord
    confirmed: bool  # this face's person_id already is the searched-for person
    similarity: float  # 1.0 for confirmed hits (a real link, not a guess)


def search_photo(
    catalog: FaceCatalog,
    gallery: PersonGallery,
    person: Person,
    path: Path,
    min_similarity: float,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[SearchHit]:
    """Return `person`'s occurrences in a single photo, in whatever order its
    face records happen to be in (not sorted) -- the shared per-photo unit
    both `search_person` and the UI's progressive `FolderSearchWorker` build
    on, so a folder scan can display/report results photo by photo instead
    of only after every photo has been processed.

    Detector boxes below `min_confidence`, and dismissed ones, are never
    considered, matching what's normally shown for that photo. A face
    already confirmed as *someone else* is never offered as a candidate
    match, no matter how similar its embedding happens to be -- that link is
    already known, not a guess to second-guess.
    """
    hits = []
    for record in catalog.visible_faces(path, min_confidence=min_confidence):
        if record.person_id == person.id:
            hits.append(SearchHit(path=path, record=record, confirmed=True, similarity=1.0))
            continue
        if record.person_id is not None:
            continue  # confirmed as a different person -- not a candidate for this search
        similarity = gallery.similarity_to(person.id, record.embedding)
        if similarity >= min_similarity:
            hits.append(SearchHit(path=path, record=record, confirmed=False, similarity=similarity))
    return hits


def search_person(
    catalog: FaceCatalog,
    gallery: PersonGallery,
    person: Person,
    paths: list[Path],
    min_similarity: float,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[SearchHit]:
    """Scan `paths` (photos in a folder) for `person`, using each photo's
    cached (or freshly detected, via `catalog.visible_faces`) face records.

    Confirmed hits (a face already labeled as `person`) sort first; below
    them, unconfirmed-but-similar hits follow, ranked by similarity,
    descending. This is the whole-folder, fully-sorted result -- for a large
    folder of not-yet-processed photos, `FolderSearchWorker` reports results
    photo by photo instead of making the caller wait for this to return.
    """
    hits = [
        hit
        for path in paths
        for hit in search_photo(catalog, gallery, person, path, min_similarity, min_confidence)
    ]
    confirmed = [hit for hit in hits if hit.confirmed]
    unconfirmed = sorted((hit for hit in hits if not hit.confirmed), key=lambda hit: hit.similarity, reverse=True)
    return confirmed + unconfirmed
