from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, Qt
from PySide6.QtGui import QImage, QMouseEvent, QResizeEvent, QWheelEvent

from tamis.views.image_viewer import ImageViewer


def _make_qimage(w: int = 200, h: int = 100) -> QImage:
    image = QImage(w, h, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.darkGray)
    return image


def _wheel_event(delta_y: int = 120) -> QWheelEvent:
    return QWheelEvent(
        QPointF(50, 50),
        QPointF(50, 50),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _resize_event(size: QSize) -> QResizeEvent:
    return QResizeEvent(size, QSize(size.width() - 10, size.height() - 10))


def _double_click_event(pos: QPointF = QPointF(50, 50)) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _mouse_event(event_type: QEvent.Type, pos: QPointF, button=Qt.MouseButton.LeftButton) -> QMouseEvent:
    buttons = button if event_type != QEvent.Type.MouseMove else Qt.MouseButton.LeftButton
    return QMouseEvent(event_type, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier)


def _drag(viewer: ImageViewer, start: QPointF, end: QPointF) -> None:
    viewer.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, start))
    viewer.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, end))
    viewer.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, end))


def test_crop_drag_emits_crop_selected_and_leaves_the_rect_visible(qapp):
    # Regression test for the shared _RubberBandDrawer extraction: crop's
    # drawn rect must deliberately survive mouseReleaseEvent (the caller
    # clears it later via set_crop_mode(False) once it's handled
    # crop_selected), unlike a manually-drawn face box.
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())
    viewer.set_crop_mode(True)

    received = []
    viewer.crop_selected.connect(lambda box: received.append(box))

    _drag(viewer, QPointF(10, 10), QPointF(60, 40))

    assert len(received) == 1
    left, top, right, bottom = received[0]
    assert right > left and bottom > top
    assert viewer._crop_drawer.item is not None, "the crop rect should still be visible after release"


def test_face_box_drag_emits_face_box_added_and_clears_immediately(qapp):
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())
    viewer.set_face_edit_mode(True)

    received = []
    viewer.face_box_added.connect(lambda box: received.append(box))

    _drag(viewer, QPointF(10, 10), QPointF(60, 40))

    assert len(received) == 1
    left, top, right, bottom = received[0]
    assert right > left and bottom > top
    assert viewer._face_drawer.item is None, "a drawn face box should clear itself immediately after release"


def test_a_degenerate_drag_emits_no_signal(qapp):
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())
    viewer.set_crop_mode(True)

    received = []
    viewer.crop_selected.connect(lambda box: received.append(box))

    _drag(viewer, QPointF(10, 10), QPointF(10, 10))  # zero-size

    assert received == []


def test_clicking_an_existing_face_box_dismisses_it_instead_of_drawing(qapp):
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())
    viewer.set_face_boxes([(5, 5, 50, 50)])
    viewer.set_face_edit_mode(True)

    dismissed = []
    added = []
    viewer.face_box_dismiss_requested.connect(lambda index: dismissed.append(index))
    viewer.face_box_added.connect(lambda box: added.append(box))

    inside_point = viewer.mapFromScene(20, 20)
    viewer.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, QPointF(inside_point)))

    assert dismissed == [0]
    assert added == []
    assert viewer._face_drawer.item is None, "clicking an existing box must not also start a new draw"


def test_resize_no_longer_refits_after_a_manual_zoom(qapp):
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())
    fitted_transform = viewer.transform()

    viewer.wheelEvent(_wheel_event())
    assert viewer._user_zoomed is True
    zoomed_transform = viewer.transform()
    assert zoomed_transform != fitted_transform

    # A viewport resize -- e.g. from nudging the side-panel splitter, not
    # just resizing the window -- must not silently discard the zoom.
    viewer.resizeEvent(_resize_event(QSize(320, 220)))
    assert viewer.transform() == zoomed_transform


def test_resize_still_refits_when_the_user_has_not_zoomed(qapp):
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())

    # No manual zoom -- ordinary resize behavior (auto-fit) must be unchanged.
    viewer.resizeEvent(_resize_event(QSize(150, 100)))
    assert viewer._user_zoomed is False


def test_a_manual_zoom_survives_navigating_to_the_next_photo(qapp):
    # This used to reset. Culling means comparing the same detail across a
    # burst, which is impossible if every navigation returns to fit and you
    # have to re-zoom and re-pan for each frame.
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())
    viewer.wheelEvent(_wheel_event())
    assert viewer._user_zoomed is True
    zoomed = viewer.transform().m11()

    viewer.set_image(_make_qimage())  # e.g. navigating to the next photo

    assert viewer._user_zoomed is True
    assert abs(viewer.transform().m11() - zoomed) < 1e-6


def test_double_click_resets_zoom_to_fit(qapp):
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())
    fitted_transform = viewer.transform()

    viewer.wheelEvent(_wheel_event())
    assert viewer.transform() != fitted_transform

    viewer.mouseDoubleClickEvent(_double_click_event())
    assert viewer._user_zoomed is False
    assert viewer.transform() == fitted_transform


def _viewer(qapp, w: int = 400, h: int = 300) -> ImageViewer:
    viewer = ImageViewer()
    viewer.resize(320, 240)
    viewer.set_image(_make_qimage(w, h))
    return viewer


def _relative_center(viewer: ImageViewer) -> tuple[float, float]:
    pixmap = viewer._pixmap_item.pixmap()
    center = viewer.mapToScene(viewer.viewport().rect().center())
    return center.x() / pixmap.width(), center.y() / pixmap.height()


def test_a_new_photo_fits_the_window_when_the_user_has_not_zoomed(qapp):
    viewer = _viewer(qapp)
    fitted = viewer.transform().m11()
    viewer.set_image(_make_qimage(800, 600))
    # Refitted for the new size rather than carrying the old scale over.
    assert viewer.transform().m11() != fitted
    assert not viewer._user_zoomed


def test_zoom_carries_over_to_the_next_photo(qapp):
    """The whole point of zooming while culling is comparing the same detail
    across a burst. Resetting to fit on every navigation made that impossible:
    you had to re-zoom and re-pan for every frame."""
    viewer = _viewer(qapp)
    viewer.toggle_actual_size()
    assert viewer.is_actual_size()

    viewer.set_image(_make_qimage(400, 300))

    assert viewer.is_actual_size()
    assert viewer._user_zoomed


def test_the_relative_position_is_kept_across_a_different_shaped_photo(qapp):
    # Relative, not absolute: the next photo may be a different size or
    # orientation, and the same pixel coordinates could fall outside it.
    # Images much larger than the viewport, so there is room to pan: Qt clamps
    # centerOn to keep the viewport inside the scene, and a barely-oversized
    # image cannot be positioned freely enough to test this.
    viewer = _viewer(qapp, 2000, 1500)
    viewer.toggle_actual_size()
    pixmap = viewer._pixmap_item.pixmap()
    viewer.centerOn(pixmap.width() * 0.25, pixmap.height() * 0.75)
    before = _relative_center(viewer)

    viewer.set_image(_make_qimage(1500, 2000))  # portrait instead of landscape

    after = _relative_center(viewer)
    assert abs(after[0] - before[0]) < 0.02
    assert abs(after[1] - before[1]) < 0.02


def test_a_centre_panned_outside_the_image_is_clamped(qapp):
    # Panning can leave the viewport centre off the image; carrying that over
    # would open the next photo showing empty space.
    viewer = _viewer(qapp)
    viewer.toggle_actual_size()
    viewer.centerOn(-5000, -5000)

    viewer.set_image(_make_qimage(400, 300))

    x, y = _relative_center(viewer)
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_toggling_actual_size_returns_to_fit(qapp):
    viewer = _viewer(qapp)
    fitted = viewer.transform().m11()

    viewer.toggle_actual_size()
    assert viewer.is_actual_size()

    viewer.toggle_actual_size()
    assert abs(viewer.transform().m11() - fitted) < 1e-6
    assert not viewer._user_zoomed


def test_zoom_controls_do_nothing_without_an_image(qapp):
    viewer = ImageViewer()
    viewer.toggle_actual_size()  # must not raise
    viewer.set_image(QImage())
    assert not viewer._user_zoomed


def test_the_shortcuts_dialog_documents_zoom():
    # It was entirely undiscoverable: wheel-zoom and double-click-to-fit
    # existed but appeared nowhere in the app.
    import tamis.main_window as mw_module

    assert "Zoom" in mw_module.SHORTCUTS_TEXT
    assert "1:1" in mw_module.SHORTCUTS_TEXT
