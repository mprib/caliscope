"""Display-specific undistortion for GUI frame rendering.

This module provides coordinated frame and point transformation so that
detected points overlay correctly on undistorted frames displayed to users.
"""

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraData

logger = logging.getLogger(__name__)


class CameraUndistortView:
    """Handles undistortion for display purposes.

    Provides coordinated frame and point transformation so that points
    overlay correctly on undistorted frames. Uses precomputed remap tables
    for efficient frame undistortion with correct output sizing.

    This is a presentation-layer class. Domain-level undistortion (for
    triangulation/bundle adjustment) uses CameraData.undistort_points directly.
    """

    def __init__(self, camera: CameraData, size: tuple[int, int], scale_factor: float = 1.0):
        """Initialize the undistort view.

        Args:
            camera: CameraData with calibrated intrinsics
            size: (height, width) of the source images
            scale_factor: Display scaling factor (1.0 = fit all undistorted content,
                          >1.0 = zoom out to show more border area,
                          <1.0 = zoom in / crop)
        """
        self._camera = camera
        self._size = size
        self._scale_factor = scale_factor

        # Remap tables for efficient undistortion
        self._map1: NDArray | None = None
        self._map2: NDArray | None = None
        self._new_matrix: NDArray | None = None
        self._output_size: tuple[int, int] | None = None  # (width, height)

        self._compute_display_params()

    @property
    def is_ready(self) -> bool:
        """Check if the view has valid display parameters."""
        return self._map1 is not None

    @property
    def output_size(self) -> tuple[int, int] | None:
        """Output frame dimensions (width, height) after undistortion."""
        return self._output_size

    def set_scale_factor(self, scale_factor: float) -> None:
        """Update the scale factor and recompute display parameters."""
        self._scale_factor = scale_factor
        self._compute_display_params()

    def _compute_display_params(self) -> None:
        """Compute remap tables for undistorted display.

        Uses initUndistortRectifyMap + remap instead of cv2.undistort to allow
        output size different from input size. This ensures all undistorted
        content is visible (barrel distortion expands edges outward).
        """
        if self._camera.matrix is None or self._camera.distortions is None:
            logger.debug(f"Camera {self._camera.port} lacks calibration; cannot compute display params")
            return

        h, w = self._size
        matrix = self._camera.matrix
        distortions = self._camera.distortions

        # Get initial new camera matrix that retains all source pixels
        # Note: fisheye and standard models have different functions for this
        if self._camera.fisheye:
            # balance=1.0 is analogous to alpha=1 (retain all source pixels)
            self._new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                matrix, distortions, (w, h), np.eye(3), balance=1.0
            )
        else:
            self._new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortions, (w, h), 1, (w, h))

        # Sample points around perimeter to find actual bounds after undistortion
        # Use valid pixel coordinates [0, w-1] and [0, h-1]
        edge_samples = 20
        top = np.column_stack([np.linspace(0, w - 1, edge_samples), np.zeros(edge_samples)])
        bottom = np.column_stack([np.linspace(0, w - 1, edge_samples), np.full(edge_samples, h - 1)])
        left = np.column_stack([np.zeros(edge_samples), np.linspace(0, h - 1, edge_samples)])
        right = np.column_stack([np.full(edge_samples, w - 1), np.linspace(0, h - 1, edge_samples)])
        perimeter_points = np.vstack([top, bottom, left, right]).astype(np.float32)

        # Undistort perimeter points - must use matching distortion model
        if self._camera.fisheye:
            undistorted_pts = cv2.fisheye.undistortPoints(
                perimeter_points.reshape(-1, 1, 2), matrix, distortions, P=self._new_matrix
            )
        else:
            undistorted_pts = cv2.undistortPoints(
                perimeter_points.reshape(-1, 1, 2), matrix, distortions, P=self._new_matrix
            )

        # Find bounds of undistorted content
        min_x = float(np.min(undistorted_pts[:, 0, 0]))
        max_x = float(np.max(undistorted_pts[:, 0, 0]))
        min_y = float(np.min(undistorted_pts[:, 0, 1]))
        max_y = float(np.max(undistorted_pts[:, 0, 1]))

        # Calculate output dimensions to contain all content
        content_width = max_x - min_x
        content_height = max_y - min_y

        # Apply scale factor with sanity bounds
        max_scale = 3.0  # Don't let output exceed 3x input size
        output_w = min(int(content_width * self._scale_factor), int(w * max_scale))
        output_h = min(int(content_height * self._scale_factor), int(h * max_scale))

        # Ensure minimum size
        output_w = max(output_w, 1)
        output_h = max(output_h, 1)

        self._output_size = (output_w, output_h)

        # Recompute new_matrix for the actual output size
        if self._camera.fisheye:
            self._new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                matrix, distortions, (w, h), np.eye(3), balance=1.0, new_size=(output_w, output_h)
            )
        else:
            self._new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortions, (w, h), 1, (output_w, output_h))

        # Build remap tables for efficient frame processing
        # CV_16SC2 uses fixed-point arithmetic - faster than CV_32FC1 for display
        if self._camera.fisheye:
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                matrix, distortions, np.eye(3), self._new_matrix, (output_w, output_h), cv2.CV_16SC2
            )
        else:
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                matrix, distortions, np.eye(3), self._new_matrix, (output_w, output_h), cv2.CV_16SC2
            )

        logger.debug(
            f"Display undistort for port {self._camera.port}: "
            f"input {(w, h)} → output {self._output_size}, scale={self._scale_factor}"
        )

    def undistort_frame(self, frame: NDArray) -> NDArray:
        """Undistort a frame for display.

        Uses precomputed remap tables for efficiency. Output size may differ
        from input to ensure all undistorted content is visible.

        Args:
            frame: Input image from camera

        Returns:
            Undistorted frame with correct output sizing
        """
        if self._map1 is None or self._map2 is None:
            logger.warning(f"Cannot undistort frame for port {self._camera.port}: no display params")
            return frame

        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

    def transform_points(self, points: NDArray) -> NDArray:
        """Transform points to match undistorted frame coordinates.

        Uses the same new_matrix as frame undistortion so points align
        correctly when overlaid on the undistorted frame.

        Args:
            points: (N, 2) array of distorted pixel coordinates

        Returns:
            (N, 2) array of points in undistorted display coordinates
        """
        if self._new_matrix is None:
            logger.warning(f"Cannot transform points for port {self._camera.port}: no display params")
            return points

        if self._camera.matrix is None or self._camera.distortions is None:
            return points

        # Reshape for OpenCV
        points_reshaped = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)

        # Undistort using the display new_matrix so points align with undistorted frame
        if self._camera.fisheye:
            undistorted = cv2.fisheye.undistortPoints(
                points_reshaped, self._camera.matrix, self._camera.distortions, P=self._new_matrix
            )
        else:
            undistorted = cv2.undistortPoints(
                points_reshaped, self._camera.matrix, self._camera.distortions, P=self._new_matrix
            )

        return undistorted.reshape(-1, 2)
