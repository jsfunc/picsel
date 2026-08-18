from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")  # FaceCatalog transitively needs it; see requirements-recognition.txt

from PIL import Image  # noqa: E402

from picsel.recognition.faces import FaceCatalog  # noqa: E402
from picsel.recognition.gallery import PersonGallery  # noqa: E402
from picsel.recognition.search import search_person, search_photo  # noqa: E402


def _make_image(path: Path) -> None:
    Image.new("RGB", (200, 200), (100, 120, 140)).save(path)


def _seed_face(catalog: FaceCatalog, path: Path, box, embedding, *, person_id=None, confidence=0.99):
    """Inject a fully-formed FaceRecord directly, bypassing real detection
    (a blank synthetic image has no real face for MTCNN to find)."""
    from picsel.recognition.faces import FaceRecord

    record = FaceRecord(box=box, confidence=confidence, embedding=embedding, person_id=person_id)
    catalog.faces_for(path)  # ensure the (empty, since no real face) list exists first
    catalog._records[path.name].append(record)
    return record


def test_search_confirmed_hits_come_before_unconfirmed(tmp_path):
    photo_a = tmp_path / "a.jpg"
    photo_b = tmp_path / "b.jpg"
    _make_image(photo_a)
    _make_image(photo_b)

    gallery = PersonGallery(path=tmp_path / "people.json.gz")
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    base = rng.normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, base)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    # An unconfirmed-but-similar face in photo_b, and a confirmed one in photo_a.
    similar = base + rng.normal(scale=0.01, size=512).astype(np.float32)
    _seed_face(catalog, photo_b, (0, 0, 50, 50), similar, person_id=None)
    _seed_face(catalog, photo_a, (0, 0, 50, 50), base, person_id=person.id)

    hits = search_person(catalog, gallery, person, [photo_a, photo_b], min_similarity=0.5)

    assert len(hits) == 2
    assert hits[0].confirmed is True
    assert hits[0].path == photo_a
    assert hits[1].confirmed is False
    assert hits[1].path == photo_b


def test_search_excludes_matches_below_min_similarity(tmp_path):
    photo = tmp_path / "a.jpg"
    _make_image(photo)

    gallery = PersonGallery(path=tmp_path / "people.json.gz")
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    gallery.add_embedding(person.id, rng.normal(size=512).astype(np.float32))

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    unrelated = rng.normal(size=512).astype(np.float32)
    _seed_face(catalog, photo, (0, 0, 50, 50), unrelated)

    hits = search_person(catalog, gallery, person, [photo], min_similarity=0.9)
    assert hits == []


def test_search_excludes_dismissed_and_low_confidence_faces(tmp_path):
    photo = tmp_path / "a.jpg"
    _make_image(photo)

    gallery = PersonGallery(path=tmp_path / "people.json.gz")
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    base = rng.normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, base)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    dismissed = _seed_face(catalog, photo, (0, 0, 50, 50), base, person_id=person.id, confidence=0.99)
    dismissed.dismissed = True
    _seed_face(catalog, photo, (60, 60, 100, 100), base, person_id=person.id, confidence=0.1)

    hits = search_person(catalog, gallery, person, [photo], min_similarity=0.5, min_confidence=0.9)
    assert hits == []


def test_search_multiple_unconfirmed_hits_sorted_by_similarity(tmp_path):
    photo_a = tmp_path / "a.jpg"
    photo_b = tmp_path / "b.jpg"
    _make_image(photo_a)
    _make_image(photo_b)

    gallery = PersonGallery(path=tmp_path / "people.json.gz")
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    base = rng.normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, base)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    closer = base + rng.normal(scale=0.01, size=512).astype(np.float32)
    farther = base + rng.normal(scale=0.05, size=512).astype(np.float32)
    _seed_face(catalog, photo_a, (0, 0, 50, 50), farther)
    _seed_face(catalog, photo_b, (0, 0, 50, 50), closer)

    hits = search_person(catalog, gallery, person, [photo_a, photo_b], min_similarity=0.0)

    assert [hit.path for hit in hits] == [photo_b, photo_a]
    assert hits[0].similarity > hits[1].similarity


def test_search_excludes_faces_already_confirmed_as_someone_else(tmp_path):
    # Thomas's face happens to be quite similar to Alice's embedding (family
    # resemblance, embedding noise, whatever) -- but since it's already
    # confirmed as Thomas, it must never show up as a candidate for Alice.
    photo = tmp_path / "DSC_0035.jpg"
    _make_image(photo)

    gallery = PersonGallery(path=tmp_path / "people.json.gz")
    alice = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    alice_embedding = rng.normal(size=512).astype(np.float32)
    gallery.add_embedding(alice.id, alice_embedding)
    thomas = gallery.add_person("Thomas")

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    thomas_face = alice_embedding + rng.normal(scale=0.01, size=512).astype(np.float32)  # very similar to Alice
    _seed_face(catalog, photo, (0, 0, 50, 50), thomas_face, person_id=thomas.id)

    hits = search_person(catalog, gallery, alice, [photo], min_similarity=0.0)

    assert hits == []


def test_search_photo_returns_hits_for_just_that_photo(tmp_path):
    photo_a = tmp_path / "a.jpg"
    photo_b = tmp_path / "b.jpg"
    _make_image(photo_a)
    _make_image(photo_b)

    gallery = PersonGallery(path=tmp_path / "people.json.gz")
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    base = rng.normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, base)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    _seed_face(catalog, photo_a, (0, 0, 50, 50), base, person_id=person.id)
    _seed_face(catalog, photo_b, (0, 0, 50, 50), base, person_id=person.id)

    hits = search_photo(catalog, gallery, person, photo_a, min_similarity=0.5)

    assert len(hits) == 1
    assert hits[0].path == photo_a


def test_search_person_is_consistent_with_search_photo_per_photo(tmp_path):
    # search_person should just be search_photo over every path, confirmed
    # first then unconfirmed sorted -- not some independently-drifting logic.
    photo_a = tmp_path / "a.jpg"
    photo_b = tmp_path / "b.jpg"
    _make_image(photo_a)
    _make_image(photo_b)

    gallery = PersonGallery(path=tmp_path / "people.json.gz")
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    base = rng.normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, base)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    _seed_face(catalog, photo_a, (0, 0, 50, 50), base + rng.normal(scale=0.05, size=512).astype(np.float32))
    _seed_face(catalog, photo_b, (0, 0, 50, 50), base, person_id=person.id)

    combined = search_person(catalog, gallery, person, [photo_a, photo_b], min_similarity=0.0)
    per_photo = search_photo(catalog, gallery, person, photo_a, min_similarity=0.0) + search_photo(
        catalog, gallery, person, photo_b, min_similarity=0.0
    )

    assert {hit.path for hit in combined} == {hit.path for hit in per_photo}
    assert combined[0].confirmed is True  # search_person still sorts confirmed first
