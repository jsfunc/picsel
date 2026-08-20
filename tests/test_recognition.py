import pytest

torch = pytest.importorskip("torch")  # recognition deps are optional; see requirements-recognition.txt

from PIL import Image  # noqa: E402

from tamis.recognition import FaceDetection, detect_faces, embed_faces  # noqa: E402
from tamis.recognition.embedder import EMBEDDING_SIZE  # noqa: E402


def test_detect_faces_on_blank_image_finds_nothing():
    image = Image.new("RGB", (200, 200), (120, 150, 180))
    assert detect_faces(image) == []


def test_embed_faces_on_empty_detections_returns_empty_without_running_the_model():
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    assert embed_faces(image, []) == []


def test_embed_faces_returns_one_embedding_per_detection():
    image = Image.new("RGB", (200, 200), (100, 120, 140))
    detections = [
        FaceDetection(box=(20, 20, 100, 100), confidence=0.99),
        FaceDetection(box=(80, 80, 180, 180), confidence=0.87),
    ]

    embeddings = embed_faces(image, detections)

    assert len(embeddings) == 2
    for embedding in embeddings:
        assert embedding.shape == (EMBEDDING_SIZE,)


def test_embed_faces_is_deterministic_in_eval_mode():
    image = Image.new("RGB", (200, 200), (60, 90, 200))
    detections = [FaceDetection(box=(20, 20, 150, 150), confidence=0.99)]

    first = embed_faces(image, detections)[0]
    second = embed_faces(image, detections)[0]

    assert (first == second).all()
