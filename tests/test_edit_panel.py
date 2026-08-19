import time

from picsel.views.edit_panel import SLIDER_DEFAULT, EditPanel


def test_keyboard_driven_slider_change_eventually_commits(qapp):
    # Regression test: brightness/contrast/saturation only committed to undo
    # history via sliderReleased, which fires only after a mouse-drag
    # release. Arrow keys/Page Up-Down/Home-End (valid QSlider interactions,
    # and the only path for keyboard-only or assistive-tech use) change the
    # value via valueChanged alone -- setValue() below simulates exactly
    # that code path (no mouse press/release involved either way).
    panel = EditPanel()

    committed = []
    panel.adjustments_committed.connect(lambda: committed.append(1))

    panel.brightness_slider.setValue(SLIDER_DEFAULT + 20)
    assert committed == [], "must not commit synchronously on the value change itself"
    assert panel._commit_timer.isActive()

    deadline = time.time() + 3
    while panel._commit_timer.isActive() and time.time() < deadline:
        time.sleep(0.02)
        qapp.processEvents()

    assert committed == [1]


def test_slider_released_commits_immediately_and_cancels_the_fallback_timer(qapp):
    panel = EditPanel()

    committed = []
    panel.adjustments_committed.connect(lambda: committed.append(1))

    panel.brightness_slider.setValue(SLIDER_DEFAULT + 20)
    panel._on_slider_released()  # simulates the mouse-drag-release path

    assert committed == [1]
    assert not panel._commit_timer.isActive()


def test_reset_adjustment_sliders_does_not_start_the_commit_timer(qapp):
    panel = EditPanel()
    panel.brightness_slider.setValue(SLIDER_DEFAULT + 20)
    panel._commit_timer.stop()  # pretend the earlier change already settled

    panel.reset_adjustment_sliders()

    assert not panel._commit_timer.isActive()
