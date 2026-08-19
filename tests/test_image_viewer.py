from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, Qt
from PySide6.QtGui import QImage, QMouseEvent, QResizeEvent, QWheelEvent

from picsel.views.image_viewer import ImageViewer


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


def test_set_image_resets_the_manual_zoom_flag(qapp):
    viewer = ImageViewer()
    viewer.resize(300, 200)
    viewer.set_image(_make_qimage())
    viewer.wheelEvent(_wheel_event())
    assert viewer._user_zoomed is True

    viewer.set_image(_make_qimage())  # e.g. navigating to the next photo
    assert viewer._user_zoomed is False


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
