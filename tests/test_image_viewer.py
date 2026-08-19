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
