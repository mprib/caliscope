"""Thumbnail display with rotation control for a single camera.

Displays a camera's thumbnail image with a rotation button. Emits signals
for rotation requests — does not interact with presenter directly.
"""

from typing import TYPE_CHECKING

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from numpy.typing import NDArray

from caliscope import ICONS_DIR
from caliscope.gui.frame_emitters.tools import apply_rotation, cv2_to_qlabel

if TYPE_CHECKING:
    from caliscope.packets import PointPacket


class CameraThumbnailCard(QFrame):
    """Card displaying camera thumbnail with rotation controls.

    Signals:
        rotate_requested: Emitted when user clicks rotate.
            Payload is (port, direction) where direction is +1 (CW) or -1 (CCW).
    """

    rotate_requested = Signal(int, int)  # (port, direction)

    THUMBNAIL_SIZE = 200  # pixels
    ICON_SIZE = 24  # pixels for rotation buttons

    def __init__(self, port: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._port = port
        self._rotation_count = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the card layout."""
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        layout = QVBoxLayout(self)

        # Port label
        self._port_label = QLabel(f"Port {self._port}")
        self._port_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._port_label)

        # Thumbnail display
        self._thumbnail_label = QLabel()
        self._thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail_label.setMinimumSize(self.THUMBNAIL_SIZE, self.THUMBNAIL_SIZE)
        self._thumbnail_label.setStyleSheet("background-color: #1a1a1a;")
        layout.addWidget(self._thumbnail_label)

        # Compact rotation controls with icons
        rotation_row = QHBoxLayout()
        rotation_row.setContentsMargins(0, 4, 0, 0)
        rotation_row.addStretch()

        self._rotate_left_btn = QToolButton()
        self._rotate_left_btn.setIcon(QIcon(str(ICONS_DIR / "rotate-camera-left.svg")))
        self._rotate_left_btn.setToolTip("Rotate counter-clockwise")
        self._rotate_left_btn.setFixedSize(self.ICON_SIZE + 4, self.ICON_SIZE + 4)
        self._rotate_left_btn.clicked.connect(lambda: self._on_rotate_clicked(-1))
        rotation_row.addWidget(self._rotate_left_btn)

        self._rotate_right_btn = QToolButton()
        self._rotate_right_btn.setIcon(QIcon(str(ICONS_DIR / "rotate-camera-right.svg")))
        self._rotate_right_btn.setToolTip("Rotate clockwise")
        self._rotate_right_btn.setFixedSize(self.ICON_SIZE + 4, self.ICON_SIZE + 4)
        self._rotate_right_btn.clicked.connect(lambda: self._on_rotate_clicked(+1))
        rotation_row.addWidget(self._rotate_right_btn)

        rotation_row.addStretch()
        layout.addLayout(rotation_row)

    @property
    def port(self) -> int:
        """Camera port for this card."""
        return self._port

    # Landmark overlay color (BGR for OpenCV)
    LANDMARK_COLOR = (0, 0, 255)  # Red
    LANDMARK_RADIUS = 3

    def set_thumbnail(
        self,
        frame: NDArray,
        rotation_count: int = 0,
        points: "PointPacket | None" = None,
    ) -> None:
        """Update the displayed thumbnail.

        Args:
            frame: BGR image from camera
            rotation_count: Current rotation (0-3) to apply for display
            points: Optional tracked landmarks to overlay as red dots
        """
        self._rotation_count = rotation_count

        # Apply rotation for display
        rotated = apply_rotation(frame, rotation_count)

        # Draw landmark overlay if points provided (View layer renders)
        if points is not None:
            rotated = self._draw_landmarks(rotated, points, rotation_count)

        # Convert to QPixmap and scale
        image = cv2_to_qlabel(rotated)
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.THUMBNAIL_SIZE,
            self.THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._thumbnail_label.setPixmap(pixmap)

    def _draw_landmarks(
        self,
        frame: NDArray,
        points: "PointPacket",
        rotation_count: int,
    ) -> NDArray:
        """Draw tracked landmarks as red dots on frame.

        Points are in original (unrotated) image coordinates, so we need to
        transform them to match the rotated display.
        """
        if len(points.point_id) == 0:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        for x, y in points.img_loc:
            # Transform point coordinates to match rotation
            rx, ry = self._rotate_point(float(x), float(y), w, h, rotation_count)
            cv2.circle(result, (int(rx), int(ry)), self.LANDMARK_RADIUS, self.LANDMARK_COLOR, -1)

        return result

    def _rotate_point(self, x: float, y: float, width: int, height: int, rotation_count: int) -> tuple[float, float]:
        """Transform point coordinates to match rotated image.

        Args:
            x, y: Original image coordinates
            width, height: Rotated image dimensions (post-rotation)
            rotation_count: Rotation (0=0°, 1=90°CW, 2=180°, 3=270°CW)

        Returns:
            Transformed (x, y) in rotated coordinate system
        """
        # Note: width/height are post-rotation, so for 90°/270° they're swapped
        if rotation_count == 0:
            return x, y
        elif rotation_count == 1:  # 90° CW
            # In 90° CW rotation: new_x = old_y, new_y = old_width - old_x
            # But we receive post-rotation dimensions, so pre-rotation width = height
            return y, height - 1 - x
        elif rotation_count == 2:  # 180°
            return width - 1 - x, height - 1 - y
        else:  # 270° CW (or 90° CCW)
            return width - 1 - y, x

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the rotation buttons."""
        self._rotate_left_btn.setEnabled(enabled)
        self._rotate_right_btn.setEnabled(enabled)

    def _on_rotate_clicked(self, direction: int) -> None:
        """Handle rotate button click.

        Args:
            direction: +1 for clockwise, -1 for counter-clockwise
        """
        self.rotate_requested.emit(self._port, direction)
