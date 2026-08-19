from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from picsel.recognition.gallery import PersonGallery  # noqa: E402
from picsel.recognition.worker import FolderSearchWorker  # noqa: E402


def test_one_photo_raising_does_not_abort_the_rest_of_the_search(monkeypatch, tmp_path, qapp):
    # Regression test (B4/B5 field-review finding): a folder search previously
    # aborted entirely the moment detection raised on a single photo partway
    # through -- everything after that photo went unsearched. It should
    # instead skip the bad photo, keep scanning, and report which photo(s)
    # failed once the whole scan finishes.
    photos = [tmp_path / f"img{i:03d}.jpg" for i in range(3)]
    gallery = PersonGallery(path=tmp_path / "people.json.gz")
    person = gallery.add_person("Alice")

    def fake_search_photo(catalog, gallery_, person_, path, min_similarity, min_confidence):
        if path == photos[1]:
            raise RuntimeError("corrupted photo")
        return [f"hit-for-{path.name}"]

    monkeypatch.setattr("picsel.recognition.worker.search_photo", fake_search_photo)

    worker = FolderSearchWorker(
        catalog=None,  # never touched directly -- search_photo is monkeypatched
        gallery=gallery,
        person=person,
        paths=photos,
        min_similarity=0.5,
    )

    processed = []
    finished = []
    worker.signals.photo_processed.connect(lambda hits, done, total: processed.append((hits, done, total)))
    worker.signals.finished.connect(lambda error, failed: finished.append((error, failed)))

    worker.run()

    assert len(processed) == 3, "the photo after the failing one must still be processed"
    assert processed[0][0] == ["hit-for-img000.jpg"]
    assert processed[1][0] == []  # the failing photo: no hits, but the scan kept going
    assert processed[2][0] == ["hit-for-img002.jpg"]

    assert len(finished) == 1
    error, failed_paths = finished[0]
    assert error == "", "a single bad photo must not be reported as a fatal search error"
    assert failed_paths == [photos[1]]
