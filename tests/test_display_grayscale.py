import numpy as np
import pytest

from caliscope.packets import PixelFormat, PointPacket
from caliscope.recording.overlay import draw_scatter_overlay


def test_draw_scatter_overlay_grayscale_produces_visible_circles():
    """Circles drawn on grayscale frames must be visible, not black."""
    gray = np.zeros((480, 640), dtype=np.uint8)
    gray[:] = 128  # mid-gray background

    points = PointPacket(
        object_id=np.array([0], dtype=np.int32),
        keypoint_id=np.array([0], dtype=np.int32),
        img_loc=np.array([[320.0, 240.0]], dtype=np.float64),
    )

    def draw_instructions(keypoint_id):
        return {"radius": 10, "color": (0, 0, 220), "thickness": -1}

    result = draw_scatter_overlay(gray, points, draw_instructions, PixelFormat.GRAY)
    # Result should be 3-channel (converted for drawing)
    assert result.ndim == 3
    # The circle at (320, 240) should NOT be black (intensity 0)
    center_pixel = result[240, 320]
    assert center_pixel[2] > 0  # red channel should be visible


def test_draw_scatter_overlay_grayscale_no_points_still_bgr():
    """A GRAY frame with no points must still come back 3-channel (C1 guard)."""
    gray = np.full((480, 640), 128, dtype=np.uint8)
    result = draw_scatter_overlay(gray, None, None, PixelFormat.GRAY)
    assert result.ndim == 3
    assert result.shape == (480, 640, 3)


def test_cv2_to_qlabel_grayscale():
    """cv2_to_qlabel handles 2D grayscale arrays."""
    pytest.importorskip("PySide6")
    from caliscope.gui.frame_emitters.tools import cv2_to_qlabel

    gray = np.zeros((480, 640), dtype=np.uint8)
    gray[100:200, 100:200] = 128

    qimage = cv2_to_qlabel(gray)
    assert qimage.width() == 640
    assert qimage.height() == 480


def test_cv2_to_qlabel_bgr_unchanged():
    """cv2_to_qlabel still works with 3D BGR arrays."""
    pytest.importorskip("PySide6")
    from caliscope.gui.frame_emitters.tools import cv2_to_qlabel

    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    qimage = cv2_to_qlabel(bgr)
    assert qimage.width() == 640
    assert qimage.height() == 480
