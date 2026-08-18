import json
from pathlib import Path

import pytest

pytest.importorskip("torch")  # recognition deps are optional; see requirements-recognition.txt

from PIL import Image  # noqa: E402

import picsel.recognition.faces as faces_module  # noqa: E402
from picsel.recognition.detector import FaceDetection  # noqa: E402
from picsel.recognition.faces import FACES_FILENAME, FaceCatalog  # noqa: E402


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
