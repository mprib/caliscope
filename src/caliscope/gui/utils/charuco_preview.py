"""Shared charuco board preview rendering utility.

Used by ProjectSetupView and CamerasTabWidget to render charuco boards
as QPixmap without coupling the domain model to Qt more than it already is.

Note: Charuco.board_img() is a Qt-free method that generates the board
using OpenCV. This utility handles the OpenCV -> Qt conversion.
"""

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from caliscope.core.charuco import Charuco


def render_charuco_pixmap(charuco: Charuco, max_dimension: int) -> QPixmap:
    """Render charuco board as a QPixmap for preview display.

    Uses Charuco.board_img() (OpenCV) and converts to QPixmap.
    Scales to fit within max_dimension while maintaining aspect ratio.

    Args:
        charuco: The charuco board to render
        max_dimension: Maximum width or height in pixels

    Returns:
        QPixmap scaled to fit within max_dimension
    """
    # Generate board image (grayscale)
    img = charuco.board_img(pixmap_scale=max_dimension)

    # Convert grayscale to RGB for QImage
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    # Apply scaling with aspect ratio preservation and smooth transformation
    return QPixmap.fromImage(qimage.copy()).scaled(
        max_dimension,
        max_dimension,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
