import json
from pathlib import Path

import pytest

pytest.importorskip("torch")  # recognition deps are optional; see requirements-recognition.txt

from PIL import Image  # noqa: E402

import picsel.recognition.faces as faces_module  # noqa: E402
from picsel.recognition.detector import FaceDetection  # noqa: E402
from picsel.recognition.faces import FACES_FILENAME, FaceCatalog, FaceRecord  # noqa: E402


def _make_image(path: Path) -> None:
    Image.new("RGB", (200, 200), (100, 120, 140)).save(path)


def test_faces_for_runs_detection_only_once_per_image(tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    calls = []
    real_detect = faces_module.detect_faces

    def counting_detect(image):
        calls.append(1)
        return real_detect(image)

    monkeypatch.setattr(faces_module, "detect_faces", counting_detect)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    catalog.faces_for(path)
    catalog.faces_for(path)

    assert len(calls) == 1


def test_load_corrupted_state_file_sets_load_error(tmp_path):
    (tmp_path / FACES_FILENAME).write_text("not valid json{{{")

    catalog = FaceCatalog()
    catalog.load(tmp_path)

    assert catalog._records == {}
    assert catalog.load_error is not None
    assert str(tmp_path / FACES_FILENAME) in catalog.load_error


def test_load_missing_state_file_leaves_load_error_none(tmp_path):
    catalog = FaceCatalog()
    catalog.load(tmp_path)
    assert catalog.load_error is None


def test_load_error_resets_on_a_later_successful_load(tmp_path):
    other_folder = tmp_path / "other"
    other_folder.mkdir()
    (tmp_path / FACES_FILENAME).write_text("not valid json{{{")

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    assert catalog.load_error is not None

    catalog.load(other_folder)  # a folder with no sidecar at all -- perfectly normal
    assert catalog.load_error is None


class _EmbeddingWithSideEffect:
    """Wraps a real embedding array so `.tolist()` (called by `save()` while
    building its JSON-able dict) can trigger an arbitrary side effect on its
    first call -- used below to simulate a background worker inserting a new
    key into `self._records` partway through `save()`'s iteration, exactly
    the race `list(self._records.items())` guards against."""

    def __init__(self, real, on_first_call) -> None:
        self._real = real
        self._on_first_call = on_first_call

    def tolist(self):
        if self._on_first_call is not None:
            callback, self._on_first_call = self._on_first_call, None
            callback()
        return self._real.tolist()


def test_save_tolerates_a_key_inserted_mid_iteration(tmp_path):
    # Simulates a background FaceDetectionWorker/FolderSearchWorker inserting
    # a new photo's records into self._records (via faces_for()) while save()
    # is mid-iteration on the GUI thread -- previously raised "dictionary
    # changed size during iteration" (a real RuntimeError, not hypothetical,
    # since save() iterated the live dict directly).
    path_a = tmp_path / "a.jpg"
    _make_image(path_a)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    record = catalog.add_manual_face(path_a, box=(10, 10, 50, 50))

    def insert_concurrently():
        catalog._records["b.jpg"] = [record]

    record.embedding = _EmbeddingWithSideEffect(record.embedding, insert_concurrently)

    catalog.save()  # must not raise RuntimeError: dictionary changed size during iteration

    # The concurrently-inserted "b.jpg" key exists in memory (the point of
    # the race guard is "don't crash", not "capture a mutation that happens
    # after this save() already took its snapshot") -- it'll be picked up by
    # the next save().
    assert "b.jpg" in catalog._records

    reloaded = FaceCatalog()
    reloaded.load(tmp_path)
    assert set(reloaded._records.keys()) == {"a.jpg"}


def test_faces_for_does_not_cache_stale_result_after_folder_switch(tmp_path):
    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    # Same filename in both folders -- the realistic case this bug affects
    # (e.g. camera default naming like IMG_0001.jpg reused across folders).
    path_a = folder_a / "photo.jpg"
    path_b = folder_b / "photo.jpg"
    _make_image(path_a)
    _make_image(path_b)

    catalog = FaceCatalog()
    catalog.load(folder_a)

    real_detect = faces_module.detect_faces

    def switching_detect(image):
        # Simulate load() firing on the GUI thread while this (slow,
        # normally-background) detection call for folder_a is still in
        # flight -- the scenario a real FaceDetectionWorker can hit.
        catalog.load(folder_b)
        return real_detect(image)

    monkeypatch_target = faces_module.detect_faces
    faces_module.detect_faces = switching_detect
    try:
        result = catalog.faces_for(path_a)
    finally:
        faces_module.detect_faces = monkeypatch_target

    # The stale folder_a result must not have been written into folder_b's
    # cache under the shared filename "photo.jpg".
    assert "photo.jpg" not in catalog._records
    assert result == []  # still a valid (if uncached) answer for path_a itself


def test_invalidate_drops_cached_record_forcing_redetection(tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    calls = []
    real_detect = faces_module.detect_faces

    def counting_detect(image):
        calls.append(1)
        return real_detect(image)

    monkeypatch.setattr(faces_module, "detect_faces", counting_detect)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    catalog.faces_for(path)
    assert len(calls) == 1

    catalog.invalidate(path)
    catalog.faces_for(path)

    assert len(calls) == 2  # re-ran detection instead of reusing the stale cache


def test_invalidate_unknown_path_is_a_no_op(tmp_path):
    catalog = FaceCatalog()
    catalog.load(tmp_path)
    catalog.invalidate(tmp_path / "never_seen.jpg")  # must not raise


def test_add_manual_face_is_always_visible_regardless_of_threshold(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    record = catalog.add_manual_face(path, box=(20, 20, 100, 100))

    assert record.is_manual
    assert record.confidence is None
    visible = catalog.visible_faces(path, min_confidence=0.999)
    assert record in visible


def test_remove_manual_face(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    record = catalog.add_manual_face(path, box=(20, 20, 100, 100))
    catalog.remove_manual_face(path, record)

    assert record not in catalog.faces_for(path)


def test_dismiss_hides_a_detector_face_from_visible_faces(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)
    # Seed a detector-found record directly (a blank synthetic image has no
    # real face to detect), matching what `load()` would parse from a
    # previously-saved .picsel_faces.json.
    (tmp_path / FACES_FILENAME).write_text(
        json.dumps(
            {
                "photo.jpg": [
                    {"box": [10, 10, 50, 50], "confidence": 0.95, "embedding": [0.0] * 512, "dismissed": False}
                ]
            }
        )
    )

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    [record] = catalog.faces_for(path)
    assert record in catalog.visible_faces(path, min_confidence=0.5)

    catalog.dismiss(record)

    assert record not in catalog.visible_faces(path, min_confidence=0.5)
    assert record.dismissed is True


def test_visible_faces_excludes_below_threshold_detector_faces(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)
    (tmp_path / FACES_FILENAME).write_text(
        json.dumps(
            {
                "photo.jpg": [
                    {"box": [10, 10, 50, 50], "confidence": 0.95, "embedding": [0.0] * 512, "dismissed": False},
                    {"box": [60, 60, 90, 90], "confidence": 0.4, "embedding": [0.0] * 512, "dismissed": False},
                ]
            }
        )
    )

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    visible = catalog.visible_faces(path, min_confidence=0.9)

    assert len(visible) == 1
    assert visible[0].confidence == 0.95


def test_assign_person(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    record = catalog.add_manual_face(path, box=(20, 20, 100, 100))
    catalog.assign_person(record, "person-123")

    assert record.person_id == "person-123"


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    record = catalog.add_manual_face(path, box=(20, 20, 100, 100))
    catalog.assign_person(record, "person-123")
    catalog.save()

    reloaded = FaceCatalog()
    reloaded.load(tmp_path)
    [reloaded_record] = reloaded.faces_for(path)

    assert reloaded_record.box == record.box
    assert reloaded_record.confidence is None
    assert reloaded_record.person_id == "person-123"
    assert reloaded_record.embedding.shape == record.embedding.shape


def test_remap_person_updates_matching_records_only(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    record_a = catalog.add_manual_face(path, box=(10, 10, 50, 50))
    record_b = catalog.add_manual_face(path, box=(60, 60, 100, 100))
    catalog.assign_person(record_a, "old-id")
    catalog.assign_person(record_b, "unrelated-id")

    catalog.remap_person("old-id", "new-id")

    assert record_a.person_id == "new-id"
    assert record_b.person_id == "unrelated-id"  # untouched


def test_unassign_all_people_clears_every_record(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    record_a = catalog.add_manual_face(path, box=(10, 10, 50, 50))
    record_b = catalog.add_manual_face(path, box=(60, 60, 100, 100))
    catalog.assign_person(record_a, "person-1")
    catalog.assign_person(record_b, "person-2")

    catalog.unassign_all_people()

    assert record_a.person_id is None
    assert record_b.person_id is None


def test_forget_person_clears_only_that_persons_records(tmp_path):
    path = tmp_path / "photo.jpg"
    _make_image(path)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    record_a = catalog.add_manual_face(path, box=(10, 10, 50, 50))
    record_b = catalog.add_manual_face(path, box=(60, 60, 100, 100))
    catalog.assign_person(record_a, "person-1")
    catalog.assign_person(record_b, "person-2")

    catalog.forget_person("person-1")

    assert record_a.person_id is None
    assert record_b.person_id == "person-2"  # untouched


def test_face_record_equality_is_identity_not_value(tmp_path):
    # Two different records sharing box+confidence (a plausible accident --
    # e.g. two manual boxes drawn at the same spot) must not crash `in`/
    # `.remove()` by falling through to comparing their embedding arrays.
    path = tmp_path / "photo.jpg"
    _make_image(path)

    catalog = FaceCatalog()
    catalog.load(tmp_path)
    first = catalog.add_manual_face(path, box=(10, 10, 50, 50))
    second = FaceRecord(box=first.box, confidence=None, embedding=first.embedding.copy())
    records = catalog.faces_for(path)
    records.append(second)

    assert first in records
    assert second in records
    assert first is not second

    catalog.remove_manual_face(path, second)
    assert second not in records
    assert first in records
