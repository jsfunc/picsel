"""Owns aesthetic scoring for the open folder: the per-folder cache, the
background scoring pass, and its persistence.

Deliberately knows nothing about the filmstrip or the slider -- it emits a
signal when scores arrive and MainWindow decides what to redraw. Same shape
as FaceRecognitionController, including the two dedicated single-threaded
pools and the prepare/write save split.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, Signal

# Fail this module's import when the optional extra is absent, so
# MainWindow's `except ImportError` sees it and hides the whole feature.
# Checked with find_spec rather than by importing open_clip, which is slow
# enough to notice at startup and pointless for a user who never scores
# anything -- scorer.py imports it lazily for exactly that reason, which is
# also why importing *this* module alone would otherwise succeed and leave
# the score column and slider present but permanently empty.
if importlib.util.find_spec("open_clip") is None:  # pragma: no cover - depends on install
    raise ImportError("open_clip is required for quality scoring; see requirements-quality.txt")

from tamis.quality.store import QualityStore
from tamis.quality.worker import QualityScoreWorker
from tamis.recognition.worker import SaveWorker

# How many photos go to the model at once. The forward pass costs ~23ms for a
# single image and ~0.34ms per image at 16, so batching is worth roughly 70x;
# past this the gain flattens while the latency of the first visible result
# keeps growing.
BATCH_SIZE = 16


class QualityController(QObject):
    scores_updated = Signal()  # some photos got scores; the view should redraw
    progress = Signal(int, int)  # scored, total -- for the status bar
    failed = Signal(str)  # scoring broke; reported once rather than silently doing nothing

    def __init__(self, parent_widget, library) -> None:
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.library = library
        self.store = QualityStore()

        # Its own single-threaded pool, for the reasons detection has one:
        # scoring holds a thread while the GPU works, and on the shared pool
        # enough batches would occupy every thread and starve the decode of
        # the photo actually on screen.
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(1)
        self._pending: list[QualityScoreWorker] = []

        self._save_pool = QThreadPool()
        self._save_pool.setMaxThreadCount(1)
        self._pending_saves: list[SaveWorker] = []

        self._queued = 0
        self._done = 0
        self._reported_failure = False

    # -- Folder lifecycle ------------------------------------------------

    def load_folder(self, folder: Path) -> None:
        self.cancel()
        self.store.load(folder)
        names = {item.name for item in self.library.items}
        self.store.prune_to(names)

    def score_folder(self) -> None:
        """Queue every not-yet-scored photo in the open folder.

        Cheap to call repeatedly: photos already in the cache are skipped, so
        reopening a folder scores only what is new.
        """
        pending = [item.path for item in self.library.items if not self.store.has(item.path)]
        self._queued = len(pending)
        self._done = 0
        self._reported_failure = False
        if not pending:
            self.progress.emit(0, 0)
            return
        generation = self.store.generation
        for start in range(0, len(pending), BATCH_SIZE):
            self._start_batch(pending[start : start + BATCH_SIZE], generation)
        self.progress.emit(0, self._queued)

    def _start_batch(self, paths: list[Path], generation: int) -> None:
        worker = QualityScoreWorker(paths, generation)
        # Held until it finishes, same reasoning as every other worker list in
        # this app: the signals object is a QObject and can be collected
        # mid-flight if this is the only reference.
        self._pending.append(worker)
        worker.signals.finished.connect(
            lambda scores, gen, error, w=worker: self._on_batch_done(scores, gen, error, w)
        )
        self._pool.start(worker)

    def _on_batch_done(self, scores: dict, generation: int, error: str, worker) -> None:
        if worker in self._pending:
            self._pending.remove(worker)
        if worker.cancelled:
            return
        if error:
            # Report once per folder: a failure here is systemic (missing
            # weights, no disk space for the download, a broken GPU driver),
            # so repeating it for every batch would be noise.
            if not self._reported_failure:
                self._reported_failure = True
                self.failed.emit(error)
            return
        if not self.store.set_many(scores, generation):
            return  # folder changed while this batch was in flight
        self._done += len(scores)
        self.progress.emit(self._done, self._queued)
        self.scores_updated.emit()
        if not self._pending:
            self.save()

    @property
    def scoring_in_progress(self) -> bool:
        """Whether a scoring pass still has batches outstanding. Lets the UI
        explain why an order or a filter that depends on scores does not look
        right yet, instead of appearing to have ignored the request."""
        return bool(self._pending)

    @property
    def scoring_progress(self) -> tuple[int, int]:
        return self._done, self._queued

    # -- Score access ----------------------------------------------------

    def score_for(self, path: Path) -> int | None:
        return self.store.get(path)

    def invalidate(self, path: Path) -> None:
        """Called after an overwrite save: the pixels changed, so the cached
        score describes an image that no longer exists."""
        self.store.invalidate(path)

    # -- Persistence -----------------------------------------------------

    def save(self) -> None:
        prepared = self.store.prepare_save()
        if prepared is None:
            return
        path, data = prepared
        worker = SaveWorker(self.store.write_payload, path, data)
        self._pending_saves.append(worker)
        worker.signals.finished.connect(lambda _error, w=worker: self._pending_saves.remove(w)
                                        if w in self._pending_saves else None)
        self._save_pool.start(worker)

    # -- Shutdown --------------------------------------------------------

    def cancel(self) -> None:
        """Abandon queued and running scoring without waiting. Cancelled
        workers exit at the top of run(), so a queue drains in microseconds
        rather than holding the window open."""
        for worker in self._pending:
            worker.cancel()

    def wait_for_idle(self) -> None:
        self._pool.waitForDone()
        self._save_pool.waitForDone()
