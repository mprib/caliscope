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
from caliscope.task_manager.task_manager import TaskManager

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
        task_manager: TaskManager,
        depth_ratios: dict[int, float | None] | None = None,
        initial_cam_id: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cameras = cameras
        self._extrinsic_dir = extrinsic_dir
        self._task_manager = task_manager
        self._depth_ratios = depth_ratios or {}
        # First video frame per camera; None when the video could not be read.
        self._first_frames: dict[int, NDArray | None] = {}
        self._frames_loading: set[int] = set()
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

        # Depth ratio info row
        info_row = QHBoxLayout()
        self._depth_ratio_label = QLabel("Depth ratio: —")
        self._depth_ratio_label.setStyleSheet("color: #888; font-size: 11px;")
        info_row.addWidget(self._depth_ratio_label)
        info_row.addStretch()
        layout.addLayout(info_row)

        # Image labels side by side
        images_row = QHBoxLayout()
        self._before_label = QLabel("Before")
        self._before_label.setObjectName("lens_before")
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

        # Set initial camera and load images
        if initial_cam_id is not None and initial_cam_id in self._cameras:
            idx = sorted_ids.index(initial_cam_id)
            self._cam_combo.setCurrentIndex(idx)
        self._update_images()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._cam_combo.currentData() is not None:
            self._update_images()

    def _update_images(self) -> None:
        cam_id = self._cam_combo.currentData()
        if cam_id is None or cam_id not in self._cameras:
            return

        camera = self._cameras[cam_id]
        w, h = camera.size

        depth_ratio = self._depth_ratios.get(cam_id)
        if depth_ratio is not None:
            self._depth_ratio_label.setText(f"Depth ratio: {depth_ratio:.2f}×")
            self._depth_ratio_label.setToolTip(
                "Ratio of furthest to nearest triangulated 3D points (95th/5th percentile Z-depth). "
                "High values (>10×) may indicate poor conditioning for this camera's intrinsic estimation."
            )
        else:
            self._depth_ratio_label.setText("Depth ratio: —")
            self._depth_ratio_label.setToolTip("")

        use_real = self._source_combo.currentIndex() == 0
        if use_real and cam_id not in self._first_frames:
            self._request_first_frame(cam_id)
            for label in (self._before_label, self._after_label):
                label.setPixmap(QPixmap())
                label.setText("Loading frame…")
            return

        frame = self._first_frames[cam_id] if use_real else None
        if frame is None:
            frame = _generate_synthetic_grid(w, h)

        visualizer = LensModelVisualizer(camera)
        undistorted = visualizer.undistort(frame)

        available_width = max(self.width() // 2 - 24, 200)
        self._before_label.setPixmap(_ndarray_to_pixmap(frame, max_width=available_width))
        self._after_label.setPixmap(_ndarray_to_pixmap(undistorted, max_width=available_width))

    def _request_first_frame(self, cam_id: int) -> None:
        """Decode the camera's first frame in a background task."""
        if cam_id in self._frames_loading:
            return
        self._frames_loading.add(cam_id)
        extrinsic_dir = self._extrinsic_dir

        def worker(_token, _handle) -> tuple[int, NDArray | None]:
            return cam_id, _grab_first_frame(extrinsic_dir, cam_id)

        handle = self._task_manager.submit(worker, name=f"Lens model frame cam_id {cam_id}", auto_start=False)
        handle.completed.connect(self._on_first_frame_loaded, Qt.ConnectionType.QueuedConnection)
        self._task_manager.start_task(handle.task_id)

    def _on_first_frame_loaded(self, result: tuple[int, NDArray | None]) -> None:
        cam_id, frame = result
        self._frames_loading.discard(cam_id)
        self._first_frames[cam_id] = frame
        self._update_images()
