"""Shared ArUco marker preview rendering utility.

Used by ProjectSetupView and potentially other views to render
ArUco marker previews without coupling the domain model to Qt.
"""

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from caliscope.core.aruco_marker import ArucoMarkerSet


def render_aruco_pixmap(marker_set: ArucoMarkerSet, marker_id: int, size: int) -> QPixmap:
    """Render ArUco marker as a QPixmap for display.

    Uses ArucoMarkerSet.generate_marker_image() to create the annotated marker,
    then converts BGR to QPixmap. Scales to fit within target size while
    maintaining aspect ratio.
    """
    marker = marker_set.markers[marker_id]
    ppm = int(size / marker.size_m * 4.0)
    pixel_size = int(marker.size_m * ppm)
    bgr = marker_set.generate_marker_image(marker_id, pixel_size)

    # Convert BGR to RGB for Qt
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w

    # Create QImage from numpy array; .copy() ensures data ownership
    qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    pixmap = QPixmap.fromImage(qimage.copy())
    return pixmap.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
