import json
from pathlib import Path

import pytest

from tamis.quality.store import MODEL_ID, QUALITY_FILENAME, QualityStore


def _store(tmp_path) -> QualityStore:
    store = QualityStore()
    store.load(tmp_path)
    return store


def test_scores_round_trip_through_the_sidecar(tmp_path):
    store = _store(tmp_path)
    store.set_many({"a.jpg": 42, "b.jpg": 91}, store.generation)
    path, data = store.prepare_save()
    QualityStore.write_payload(path, data)

    reloaded = QualityStore()
    reloaded.load(tmp_path)

    assert reloaded.get(tmp_path / "a.jpg") == 42
    assert reloaded.get(tmp_path / "b.jpg") == 91
    assert reloaded.has(tmp_path / "a.jpg")
    assert not reloaded.has(tmp_path / "missing.jpg")


def test_results_from_a_previous_folder_are_refused(tmp_path):
    # Scoring is batched and runs in the background, so a batch can land after
    # the user has moved on. The cache is keyed by filename only, so accepting
    # it would attach one folder's score to a same-named file in another.
    store = _store(tmp_path)
    stale = store.generation
    store.load(tmp_path / "other")

    assert store.set_many({"a.jpg": 50}, stale) is False
    assert store.get(tmp_path / "a.jpg") is None


def test_prune_forgets_photos_no_longer_in_the_folder(tmp_path):
    # Otherwise a renamed-away entry could be inherited by a future file that
    # happens to reuse the name.
    store = _store(tmp_path)
    store.set_many({"a.jpg": 10, "gone.jpg": 20}, store.generation)

    store.prune_to({"a.jpg"})

    assert store.get(tmp_path / "a.jpg") == 10
    assert store.get(tmp_path / "gone.jpg") is None


def test_invalidate_drops_one_score(tmp_path):
    # Used after an overwrite save: the score describes pixels that no longer
    # exist.
    store = _store(tmp_path)
    store.set_many({"a.jpg": 77}, store.generation)
    store.invalidate(tmp_path / "a.jpg")
    assert store.get(tmp_path / "a.jpg") is None


def test_an_unreadable_sidecar_is_reported_and_starts_empty(tmp_path):
    (tmp_path / QUALITY_FILENAME).write_text("{ not json")
    store = QualityStore()
    store.load(tmp_path)
    assert store.load_error is not None
    assert store.get(tmp_path / "a.jpg") is None


def test_out_of_range_and_malformed_entries_are_ignored(tmp_path):
    # Read defensively so a hand-edited file still loads.
    (tmp_path / QUALITY_FILENAME).write_text(
        json.dumps({
            "model": MODEL_ID,
            "scores": {"ok.jpg": 55, "high.jpg": 150, "low.jpg": -3, "text.jpg": "nope"},
        })
    )
    store = QualityStore()
    store.load(tmp_path)
    assert store.get(tmp_path / "ok.jpg") == 55
    for name in ("high.jpg", "low.jpg", "text.jpg"):
        assert store.get(tmp_path / name) is None


def test_prepare_save_is_none_when_there_is_nothing_to_write(tmp_path):
    store = _store(tmp_path)
    assert store.prepare_save() is None


def test_display_score_maps_the_useful_range_onto_0_100():
    pytest.importorskip("torch")
    from tamis.quality.scorer import RAW_MAX, RAW_MIN, to_display_score

    assert to_display_score(RAW_MIN) == 0
    assert to_display_score(RAW_MAX) == 100
    assert to_display_score((RAW_MIN + RAW_MAX) / 2) == 50
    # Clamped, so an unusually high or low raw value cannot leave 0-100.
    assert to_display_score(RAW_MIN - 5) == 0
    assert to_display_score(RAW_MAX + 5) == 100
    assert to_display_score(5.0) == 50


def test_a_cancelled_batch_does_no_work():
    pytest.importorskip("torch")
    from tamis.quality.worker import QualityScoreWorker

    worker = QualityScoreWorker([Path("/nonexistent/a.jpg")], generation=1)
    worker.cancel()
    received = []
    worker.signals.finished.connect(lambda scores, gen, err: received.append((scores, gen, err)))
    worker.run()

    assert received == [({}, 1, "")]
    assert worker.cancelled


def test_scores_from_a_different_model_are_discarded(tmp_path):
    """Mixing models would order photos by two opinions at once.

    Scores are only ever compared against each other, so this has no symptom:
    the ordering is simply wrong, with nothing to notice and nothing to debug
    from. The guard is what makes changing the model a safe edit.
    """
    (tmp_path / QUALITY_FILENAME).write_text(
        json.dumps({"model": "some-older-scorer", "scores": {"a.jpg": 55}})
    )
    store = QualityStore()
    store.load(tmp_path)
    assert store.get(tmp_path / "a.jpg") is None


def test_a_sidecar_with_no_model_recorded_is_discarded(tmp_path):
    # An unlabelled file could have come from any scorer, so it is not
    # trustworthy enough to rank against current scores.
    (tmp_path / QUALITY_FILENAME).write_text(json.dumps({"a.jpg": 55, "b.jpg": 70}))
    store = QualityStore()
    store.load(tmp_path)
    assert store.get(tmp_path / "a.jpg") is None


def test_the_recorded_model_id_names_the_model_actually_used(tmp_path):
    # Guards against the two drifting apart, which is what would let stale
    # scores survive a model change.
    pytest.importorskip("torch")
    from tamis.quality import scorer

    assert scorer.CLIP_MODEL in MODEL_ID
    assert scorer.model_id() == MODEL_ID
    assert f"{scorer.RAW_MIN}-{scorer.RAW_MAX}" in MODEL_ID
