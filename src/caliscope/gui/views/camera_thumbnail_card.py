"""Thumbnail display with rotation control for a single camera.

Displays a camera's thumbnail image with a rotation button. Emits signals
for rotation requests — does not interact with presenter directly.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from numpy.typing import NDArray

from caliscope.gui.frame_emitters.tools import apply_rotation, cv2_to_qlabel


class CameraThumbnailCard(QFrame):
    """Card displaying camera thumbnail with rotation control.

    Signals:
        rotate_requested: Emitted when user clicks rotate. Payload is port.
    """

    rotate_requested = Signal(int)  # port

    THUMBNAIL_SIZE = 200  # pixels

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

        # Rotation status label
        self._rotation_label = QLabel("Rotation: 0°")
        self._rotation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._rotation_label)

        # Rotate button
        self._rotate_btn = QPushButton("Rotate 90°")
        self._rotate_btn.clicked.connect(self._on_rotate_clicked)
        layout.addWidget(self._rotate_btn)

    @property
    def port(self) -> int:
        """Camera port for this card."""
        return self._port

    def set_thumbnail(self, frame: NDArray, rotation_count: int = 0) -> None:
        """Update the displayed thumbnail.

        Args:
            frame: BGR image from camera
            rotation_count: Current rotation (0-3) to apply for display
        """
        self._rotation_count = rotation_count

        # Apply rotation for display
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

        # Update rotation label
        self._rotation_label.setText(f"Rotation: {rotation_count * 90}°")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the rotation button."""
        self._rotate_btn.setEnabled(enabled)

    def _on_rotate_clicked(self) -> None:
        """Handle rotate button click."""
        self.rotate_requested.emit(self._port)
