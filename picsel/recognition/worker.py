"""Qt async wrapper for running face detection off the UI thread.

Mirrors `picsel.thumbnails`'s ThumbnailWorker/ImageLoadWorker pattern: the
recognition package's own modules (detector, embedder, faces, gallery) stay
Qt-free, and this is the seam where that pure logic gets wired into QThreadPool.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

from picsel.recognition.detector import DEFAULT_MIN_CONFIDENCE
from picsel.recognition.faces import FaceCatalog, FaceRecord
from picsel.recognition.gallery import Person, PersonGallery
from picsel.recognition.search import search_photo


class FaceDetectionSignals(QObject):
    finished = Signal(object, list, str)  # path, list[FaceRecord], error message ("" if ok)


class FaceDetectionWorker(QRunnable):
    """Runs `catalog.faces_for(path)` (detection+embedding, or a cache hit) off the UI thread."""

    def __init__(self, catalog: FaceCatalog, path: Path) -> None:
        super().__init__()
        self.catalog = catalog
        self.path = path
        self.signals = FaceDetectionSignals()

    def run(self) -> None:
        try:
            records: list[FaceRecord] = self.catalog.faces_for(self.path)
            error = ""
        except Exception as exc:
            records = []
            error = str(exc)
        self.signals.finished.emit(self.path, records, error)


class FolderSearchSignals(QObject):
    photo_processed = Signal(list, int, int)  # list[SearchHit] found in this photo, photos done, total
    finished = Signal(str)  # error message ("" if ok)


class FolderSearchWorker(QRunnable):
    """Runs `search_photo` (which detects/embeds any not-yet-cached photo
    along the way) off the UI thread, one photo at a time, reporting results
    as each photo finishes rather than only once the whole folder is done --
    a folder with many never-before-processed photos can take a while, and
    the caller shouldn't have to wait for all of it before seeing anything."""

    def __init__(
        self,
        catalog: FaceCatalog,
        gallery: PersonGallery,
        person: Person,
        paths: list[Path],
        min_similarity: float,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        super().__init__()
        self.catalog = catalog
        self.gallery = gallery
        self.person = person
        self.paths = paths
        self.min_similarity = min_similarity
        self.min_confidence = min_confidence
        self.signals = FolderSearchSignals()
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Ask the search to stop after finishing whatever photo it's
        currently on -- thread-safe, callable from the UI thread at any time."""
        self._cancelled.set()

    def run(self) -> None:
        try:
            for index, path in enumerate(self.paths):
                if self._cancelled.is_set():
                    break
                hits = search_photo(
                    self.catalog,
                    self.gallery,
                    self.person,
                    path,
                    min_similarity=self.min_similarity,
                    min_confidence=self.min_confidence,
                )
                self.signals.photo_processed.emit(hits, index + 1, len(self.paths))
            error = ""
        except Exception as exc:
            error = str(exc)
        self.signals.finished.emit(error)
