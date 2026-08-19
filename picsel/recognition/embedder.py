"""Face embedding via a pretrained InceptionResnetV1 (VGGFace2). No app/Qt dependency."""

from __future__ import annotations

import threading

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, extract_face, fixed_image_standardization
from PIL import Image

from picsel.recognition.detector import DEVICE, FaceDetection

EMBEDDING_SIZE = 512
FACE_IMAGE_SIZE = 160  # InceptionResnetV1's expected input size
FACE_MARGIN = 20  # extra context around the tight detection box, in output-image pixels

_resnet: InceptionResnetV1 | None = None
# See detector._mtcnn_lock's comment -- same reasoning, guards lazy
# construction and the forward pass itself against concurrent worker threads.
_resnet_lock = threading.Lock()


def _get_resnet() -> InceptionResnetV1:
    global _resnet
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return _resnet


def embed_faces(image: Image.Image, detections: list[FaceDetection]) -> list[np.ndarray]:
    """Return one 512-d embedding per detection, in the same order.

    `image` must be the same image (and coordinate space) the detections' boxes were computed from.
    """
    if not detections:
        return []

    faces = [
        extract_face(image, d.box, image_size=FACE_IMAGE_SIZE, margin=FACE_MARGIN) for d in detections
    ]
    batch = fixed_image_standardization(torch.stack(faces)).to(DEVICE)
    with _resnet_lock, torch.no_grad():
        embeddings = _get_resnet()(batch)
    return list(embeddings.cpu().numpy())
