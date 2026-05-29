"""Shared charuco board preview rendering utility.

Used by ProjectSetupView and CamerasTabWidget to render charuco boards
as QPixmap without coupling the domain model to Qt more than it already is.

Note: Charuco.board_img() is a Qt-free method that generates the board
using OpenCV. This utility handles the OpenCV -> Qt conversion.
"""

import cv2
from PySide6.QtGui import QImage, QPixmap

from caliscope.core.charuco import Charuco

# Render the board well above the display box, then downscale. A high render
# starts past board_img's quasi-periodic failure band (so the retry is a no-op
# in practice) and gives a cleaner thumbnail than rendering at ~200 directly.
PREVIEW_RENDER_PX = 2000


def render_charuco_pixmap(charuco: Charuco, max_dimension: int) -> QPixmap:
    """Render charuco board as a QPixmap for preview display.

    Renders the board high (PREVIEW_RENDER_PX) via Charuco.board_img() (OpenCV),
    downscales to the display box with INTER_AREA, then converts to QPixmap.
    Aspect ratio is preserved by board_img's own width/height scaling.

    Args:
        charuco: The charuco board to render
        max_dimension: Maximum width or height in pixels for the display box

    Returns:
        QPixmap scaled to fit within max_dimension
    """
    # Generate board image high, then downscale (grayscale).
    img = charuco.board_img(pixmap_scale=PREVIEW_RENDER_PX)

    # Downscale to the display box with INTER_AREA (the OpenCV-recommended filter
    # for shrinking — avoids the moiré that bilinear/nearest produce on the
    # board's high-frequency marker grid). Aspect ratio already baked into img.
    h, w = img.shape[:2]
    scale = min(max_dimension / w, max_dimension / h)
    target_w = max(1, round(w * scale))
    target_h = max(1, round(h * scale))
    downscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Convert grayscale to RGB for QImage
    rgb = cv2.cvtColor(downscaled, cv2.COLOR_GRAY2RGB)

    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    # copy() detaches from the numpy buffer that goes out of scope after return.
    return QPixmap.fromImage(qimage.copy())
