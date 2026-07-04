"""Lens model visualization dialog.

Shows before/after undistortion for a selected camera using either
a real video frame or a synthetic grid. Uses LensModelVisualizer
for the undistortion rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from caliscope.cameras.camera_array import CameraData
from caliscope.gui.lens_model_visualizer import LensModelVisualizer
from caliscope.recording.frame_source import FrameSource

logger = logging.getLogger(__name__)


def _generate_synthetic_grid(width: int, height: int) -> NDArray:
    """Draw a regular grid of circles on a dark background."""
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    spacing_x = max(width // 20, 10)
    spacing_y = max(height // 15, 10)
    radius = max(min(spacing_x, spacing_y) // 6, 2)

    for y in range(spacing_y, height - spacing_y // 2, spacing_y):
        for x in range(spacing_x, width - spacing_x // 2, spacing_x):
            cv2.circle(frame, (x, y), radius, (200, 200, 200), -1, cv2.LINE_AA)
    return frame


def _grab_first_frame(extrinsic_dir: Path, cam_id: int) -> NDArray | None:
    """Grab the first frame from a camera's video file."""
    video_path = extrinsic_dir / f"cam_{cam_id}.mp4"
    if not video_path.exists():
        return None
    try:
        source = FrameSource(extrinsic_dir, cam_id, wanted_indices={0})
        packet = source.next_frame()
        source.close()
        return packet.frame if packet is not None else None
    except Exception:
        logger.debug(f"Could not read frame from {video_path}", exc_info=True)
        return None


def _ndarray_to_pixmap(arr: NDArray, max_width: int = 640) -> QPixmap:
    """Convert BGR NDArray to QPixmap, scaling down if wider than max_width."""
    h, w = arr.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


class LensModelDialog(QDialog):
    """Dialog showing before/after undistortion for a selected camera."""

    def __init__(
        self,
        cameras: dict[int, CameraData],
        extrinsic_dir: Path,
        initial_cam_id: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cameras = cameras
        self._extrinsic_dir = extrinsic_dir
        self.setWindowTitle("Lens Model Visualization")
        self.setModal(False)
        self.setMinimumSize(700, 400)
        self._setup_ui(initial_cam_id)

    def _setup_ui(self, initial_cam_id: int | None) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Top row: camera selector + source mode
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Camera:"))
        self._cam_combo = QComboBox()
        sorted_ids = sorted(self._cameras.keys())
        for cid in sorted_ids:
            self._cam_combo.addItem(f"Cam {cid}", cid)
        top_row.addWidget(self._cam_combo)

        top_row.addSpacing(16)
        top_row.addWidget(QLabel("Source:"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(["Real Frame", "Synthetic Grid"])
        top_row.addWidget(self._source_combo)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Image labels side by side
        images_row = QHBoxLayout()
        self._before_label = QLabel("Before")
        self._before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._before_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self._before_label.setMinimumHeight(250)

        self._after_label = QLabel("After")
        self._after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._after_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self._after_label.setMinimumHeight(250)

        images_row.addWidget(self._before_label)
        images_row.addWidget(self._after_label)
        layout.addLayout(images_row)

        # Captions
        caption_row = QHBoxLayout()
        before_cap = QLabel("Original")
        before_cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        before_cap.setStyleSheet("color: #888; font-size: 11px;")
        after_cap = QLabel("Undistorted")
        after_cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        after_cap.setStyleSheet("color: #888; font-size: 11px;")
        caption_row.addWidget(before_cap)
        caption_row.addWidget(after_cap)
        layout.addLayout(caption_row)

        # Connect signals
        self._cam_combo.currentIndexChanged.connect(self._update_images)
        self._source_combo.currentIndexChanged.connect(self._update_images)

        # Set initial camera
        if initial_cam_id is not None and initial_cam_id in self._cameras:
            idx = sorted_ids.index(initial_cam_id)
            self._cam_combo.setCurrentIndex(idx)
        else:
            self._update_images()

    def _update_images(self) -> None:
        cam_id = self._cam_combo.currentData()
        if cam_id is None or cam_id not in self._cameras:
            return

        camera = self._cameras[cam_id]
        w, h = camera.size

        use_real = self._source_combo.currentIndex() == 0
        if use_real:
            frame = _grab_first_frame(self._extrinsic_dir, cam_id)
            if frame is None:
                frame = _generate_synthetic_grid(w, h)
        else:
            frame = _generate_synthetic_grid(w, h)

        visualizer = LensModelVisualizer(camera)
        undistorted = visualizer.undistort(frame)

        self._before_label.setPixmap(_ndarray_to_pixmap(frame))
        self._after_label.setPixmap(_ndarray_to_pixmap(undistorted))
