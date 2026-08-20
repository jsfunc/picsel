import pytest

pytest.importorskip("torch")  # SearchPanel imports FolderSearchWorker, which needs it

from tamis.views.search_panel import SearchPanel  # noqa: E402


class _FakeLibrary:
    items = []


class _FakePerson:
    def __init__(self, person_id):
        self.id = person_id


class _FakeWorker:
    def __init__(self, person_id):
        self.person = _FakePerson(person_id)
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def _make_panel(qapp):
    return SearchPanel(
        library=_FakeLibrary(),
        face_catalog=None,
        person_gallery=type("G", (), {"people": []})(),
        thread_pool=None,
        get_min_confidence=lambda: 0.9,
    )


def test_cancel_if_targeting_stops_a_search_for_a_removed_person(qapp):
    # Regression test: a running FolderSearchWorker holds a direct reference
    # to the Person it's searching for. Merging or forgetting that exact
    # person via Manage People while the search is in flight previously left
    # the worker running against a Person no longer in the gallery -- it kept
    # going but silently stopped surfacing further unconfirmed hits, with no
    # indication why. Manage People now reports which ids it removed, and
    # MainWindow cancels any in-flight search targeting one of them.
    panel = _make_panel(qapp)
    worker = _FakeWorker("person-1")
    panel._worker = worker
    panel.cancel_button.setEnabled(True)

    panel.cancel_if_targeting({"person-1"})

    assert worker.cancelled is True
    assert panel._cancel_requested is True


def test_cancel_if_targeting_leaves_an_unrelated_search_running(qapp):
    panel = _make_panel(qapp)
    worker = _FakeWorker("person-1")
    panel._worker = worker
    panel.cancel_button.setEnabled(True)

    panel.cancel_if_targeting({"some-other-person"})

    assert worker.cancelled is False
    assert panel._cancel_requested is False


def test_cancel_if_targeting_is_a_no_op_with_no_search_running(qapp):
    panel = _make_panel(qapp)
    assert panel._worker is None

    panel.cancel_if_targeting({"person-1"})  # must not raise

    assert panel._worker is None
