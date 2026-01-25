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
    QSizePolicy,
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

    THUMBNAIL_SIZE = 280  # pixels (larger for better visibility)
    ICON_SIZE = 24  # pixels for rotation buttons

    def __init__(self, port: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._port = port
        self._rotation_count = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the card layout."""
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        # Prevent vertical stretching - card should be as compact as possible
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
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

    # Landmark overlay styling
    LANDMARK_COLOR = (0, 0, 255)  # Red (BGR for OpenCV)
    LANDMARK_SCALE = 0.012  # Radius as fraction of frame's smaller dimension

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

        # Draw landmark overlay BEFORE rotation - points are in original frame coordinates
        if points is not None:
            frame = self._draw_landmarks(frame, points)

        # Apply rotation for display (after drawing, so dots rotate with the image)
        rotated = apply_rotation(frame, rotation_count)

        # Convert to QPixmap and scale
        image = cv2_to_qlabel(rotated)
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.THUMBNAIL_SIZE,
            self.THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._thumbnail_label.setPixmap(pixmap)

    def _draw_landmarks(self, frame: NDArray, points: "PointPacket") -> NDArray:
        """Draw tracked landmarks as red dots on frame.

        Points from trackers are in original (unrotated) image coordinates.
        This method must be called BEFORE apply_rotation() so that the dots
        rotate together with the image content.

        Radius scales with frame size so dots remain visible after thumbnail scaling.
        """
        if len(points.point_id) == 0:
            return frame

        # Scale radius with frame size (use smaller dimension for consistency)
        h, w = frame.shape[:2]
        radius = max(3, int(min(h, w) * self.LANDMARK_SCALE))

        result = frame.copy()
        for x, y in points.img_loc:
            cv2.circle(result, (int(x), int(y)), radius, self.LANDMARK_COLOR, -1)

        return result

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
