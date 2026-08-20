"""Qt async wrapper for running face detection off the UI thread.

Mirrors `tamis.thumbnails`'s ThumbnailWorker/ImageLoadWorker pattern: the
recognition package's own modules (detector, embedder, faces, gallery) stay
Qt-free, and this is the seam where that pure logic gets wired into QThreadPool.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

from tamis.recognition.detector import DEFAULT_MIN_CONFIDENCE
from tamis.recognition.faces import FaceCatalog, FaceRecord
from tamis.recognition.gallery import Person, PersonGallery
from tamis.recognition.search import search_photo


class FaceDetectionSignals(QObject):
    finished = Signal(object, list, str)  # path, list[FaceRecord], error message ("" if ok)


class FaceDetectionWorker(QRunnable):
    """Runs `catalog.faces_for(path)` (detection+embedding, or a cache hit) off the UI thread.

    Cancellable, because these are queued faster than they can run: detection
    costs a few hundred ms per uncached photo and is serialized by the model
    lock, so a folder switch or a quit can find a substantial queue still
    waiting. `cancel()` is checked once at the start of `run()`, which is
    enough to make a queued-but-unstarted worker exit immediately -- a worker
    already past that point is left to finish, since it is a few hundred ms
    at most and its result still lands in the catalog's cache.
    """

    def __init__(self, catalog: FaceCatalog, path: Path) -> None:
        super().__init__()
        self.catalog = catalog
        self.path = path
        self.signals = FaceDetectionSignals()
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Thread-safe, callable from the UI thread at any time."""
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> None:
        if self._cancelled.is_set():
            # Still emit, so whoever is tracking pending workers can drop this
            # one; the empty record list is ignored by the generation check on
            # the receiving end.
            self.signals.finished.emit(self.path, [], "")
            return
        try:
            records: list[FaceRecord] = self.catalog.faces_for(self.path)
            error = ""
        except Exception as exc:
            records = []
            error = str(exc)
        self.signals.finished.emit(self.path, records, error)


class SaveSignals(QObject):
    finished = Signal(str)  # error message ("" if ok)


class SaveWorker(QRunnable):
    """Runs a prepared (path, data) write off the UI thread. The caller must
    have already synchronously captured a consistent snapshot via
    `FaceCatalog.prepare_save()`/`PersonGallery.prepare_save()` -- only the
    slow serialize+compress+write step happens here, so this is safe to run
    well after the snapshot was taken (a folder switch or further labeling
    on the UI thread in the meantime can't corrupt or race against what
    actually gets written, since this worker never touches the live
    catalog/gallery object at all)."""

    def __init__(self, write_payload, path: Path, data: dict) -> None:
        super().__init__()
        self._write_payload = write_payload
        self._path = path
        self._data = data
        self.signals = SaveSignals()

    def run(self) -> None:
        try:
            self._write_payload(self._path, self._data)
            error = ""
        except OSError as exc:
            error = str(exc)
        self.signals.finished.emit(error)


class FolderSearchSignals(QObject):
    photo_processed = Signal(list, int, int)  # list[SearchHit] found in this photo, photos done, total
    finished = Signal(str, list)  # fatal error message ("" if ok), list[Path] of photos that failed individually


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

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> None:
        failed: list[Path] = []
        try:
            for index, path in enumerate(self.paths):
                if self._cancelled.is_set():
                    break
                try:
                    hits = search_photo(
                        self.catalog,
                        self.gallery,
                        self.person,
                        path,
                        min_similarity=self.min_similarity,
                        min_confidence=self.min_confidence,
                    )
                except Exception:
                    # One corrupted/unreadable photo shouldn't abort every
                    # photo after it -- skip it, keep scanning, and let the
                    # caller report which ones failed once the scan finishes.
                    failed.append(path)
                    hits = []
                self.signals.photo_processed.emit(hits, index + 1, len(self.paths))
            error = ""
        except Exception as exc:
            error = str(exc)
        self.signals.finished.emit(error, failed)
