"""Visualize lens model effects for user inspection.

Shows exactly what undistortion does to a frame - no cropping or hiding
of problematic distortion. When content expands beyond the original frame,
draws a dashed boundary showing where the original frame was.
"""

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraData

logger = logging.getLogger(__name__)


def _draw_dashed_line(
    frame: NDArray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
    gap_length: int = 6,
) -> None:
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2

    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)

    if length == 0:
        return

    ux = dx / length
    uy = dy / length

    segment_length = dash_length + gap_length
    pos = 0.0

    while pos < length:
        start_x = int(x1 + ux * pos)
        start_y = int(y1 + uy * pos)

        end_pos = min(pos + dash_length, length)
        end_x = int(x1 + ux * end_pos)
        end_y = int(y1 + uy * end_pos)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)

        pos += segment_length


def _draw_dashed_rect(
    frame: NDArray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
    gap_length: int = 6,
) -> None:
    """Draw a dashed rectangle."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    _draw_dashed_line(frame, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)
    _draw_dashed_line(frame, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)
    _draw_dashed_line(frame, (x2, y2), (x1, y2), color, thickness, dash_length, gap_length)
    _draw_dashed_line(frame, (x1, y2), (x1, y1), color, thickness, dash_length, gap_length)


class LensModelVisualizer:
    """Visualizes lens model effects for user inspection.

    Shows exactly what undistortion does to a frame:
    - Content that shrinks shows black borders
    - Content that expands is fully visible with original frame boundary overlay
    - No cropping or hiding of problematic distortion

    This is a presentation-layer class. Domain-level undistortion (for
    triangulation/bundle adjustment) uses CameraData.undistort_points directly.
    """

    BOUNDARY_COLOR = (255, 255, 0)  # BGR: cyan
    BOUNDARY_THICKNESS = 2

    def __init__(self, camera: CameraData):
        """Initialize the visualizer.

        Args:
            camera: CameraData with calibrated intrinsics
        """
        self._camera = camera

        self._map1: NDArray | None = None
        self._map2: NDArray | None = None
        self._content_expands = False
        self._boundary_rect: tuple[tuple[int, int], tuple[int, int]] | None = None

        self._compute_undistortion_params()

    @property
    def is_ready(self) -> bool:
        """Check if the visualizer has valid parameters."""
        return self._map1 is not None

    @property
    def content_expands_beyond_frame(self) -> bool:
        """True if undistortion causes content to extend past original frame bounds.

        When True, undistort() draws a dashed boundary showing the original frame.
        View can use this to conditionally display a legend.
        """
        return self._content_expands

    def _compute_undistortion_params(self) -> None:
        """Compute remap tables and detect if content expands."""
        if self._camera.matrix is None or self._camera.distortions is None:
            logger.debug(f"Camera {self._camera.port} lacks calibration")
            return

        w, h = self._camera.size
        matrix = self._camera.matrix
        distortions = self._camera.distortions

        # Sample perimeter to find bounds after undistortion
        edge_samples = 20
        top = np.column_stack([np.linspace(0, w - 1, edge_samples), np.zeros(edge_samples)])
        bottom = np.column_stack([np.linspace(0, w - 1, edge_samples), np.full(edge_samples, h - 1)])
        left = np.column_stack([np.zeros(edge_samples), np.linspace(0, h - 1, edge_samples)])
        right = np.column_stack([np.full(edge_samples, w - 1), np.linspace(0, h - 1, edge_samples)])
        perimeter_points = np.vstack([top, bottom, left, right]).astype(np.float32)

        # Get new camera matrix for undistortion
        if self._camera.fisheye:
            new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                matrix, distortions, (w, h), np.eye(3), balance=1.0
            )
            undistorted_pts = cv2.fisheye.undistortPoints(
                perimeter_points.reshape(-1, 1, 2), matrix, distortions, P=new_matrix
            )
        else:
            new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortions, (w, h), 1, (w, h))
            undistorted_pts = cv2.undistortPoints(perimeter_points.reshape(-1, 1, 2), matrix, distortions, P=new_matrix)

        # Find bounds of undistorted content
        min_x = float(np.min(undistorted_pts[:, 0, 0]))
        max_x = float(np.max(undistorted_pts[:, 0, 0]))
        min_y = float(np.min(undistorted_pts[:, 0, 1]))
        max_y = float(np.max(undistorted_pts[:, 0, 1]))

        content_width = max_x - min_x
        content_height = max_y - min_y

        # Does content expand beyond original frame?
        self._content_expands = content_width > w or content_height > h

        if self._content_expands:
            # Zoom out to show all content
            scale_x = w / content_width
            scale_y = h / content_height
            scale = min(scale_x, scale_y)

            # Build scaled camera matrix centered on output
            output_center_x = w / 2
            output_center_y = h / 2

            # Where is the content center in undistorted space?
            content_center_x = (min_x + max_x) / 2
            content_center_y = (min_y + max_y) / 2

            # Build new matrix that scales and re-centers
            new_matrix_scaled = np.array(
                [
                    [matrix[0, 0] * scale, 0, output_center_x + (new_matrix[0, 2] - content_center_x) * scale],
                    [0, matrix[1, 1] * scale, output_center_y + (new_matrix[1, 2] - content_center_y) * scale],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )

            # Where do the original frame corners land?
            original_corners = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.float32)

            if self._camera.fisheye:
                corners_undist = cv2.fisheye.undistortPoints(original_corners, matrix, distortions, P=new_matrix_scaled)
            else:
                corners_undist = cv2.undistortPoints(original_corners, matrix, distortions, P=new_matrix_scaled)

            corners = corners_undist.reshape(-1, 2)
            bx1 = int(np.min(corners[:, 0]))
            bx2 = int(np.max(corners[:, 0]))
            by1 = int(np.min(corners[:, 1]))
            by2 = int(np.max(corners[:, 1]))
            self._boundary_rect = ((bx1, by1), (bx2, by2))

            # Build remap tables
            if self._camera.fisheye:
                self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                    matrix, distortions, np.eye(3), new_matrix_scaled, (w, h), cv2.CV_16SC2
                )
            else:
                self._map1, self._map2 = cv2.initUndistortRectifyMap(
                    matrix, distortions, np.eye(3), new_matrix_scaled, (w, h), cv2.CV_16SC2
                )

            logger.debug(
                f"LensModelVisualizer port {self._camera.port}: content expands, "
                f"scale={scale:.3f}, boundary={self._boundary_rect}"
            )
        else:
            # Content fits or shrinks - just undistort, black borders will appear naturally
            self._boundary_rect = None

            if self._camera.fisheye:
                final_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    matrix, distortions, (w, h), np.eye(3), balance=1.0, new_size=(w, h)
                )
                self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                    matrix, distortions, np.eye(3), final_matrix, (w, h), cv2.CV_16SC2
                )
            else:
                final_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortions, (w, h), 1, (w, h))
                self._map1, self._map2 = cv2.initUndistortRectifyMap(
                    matrix, distortions, np.eye(3), final_matrix, (w, h), cv2.CV_16SC2
                )

            logger.debug(f"LensModelVisualizer port {self._camera.port}: content fits within frame")

    def undistort(self, frame: NDArray) -> NDArray:
        """Undistort a frame for visualization.

        If content expands beyond the original frame, draws a dashed boundary
        showing where the original frame was.

        Args:
            frame: Input image (possibly with composited overlays)

        Returns:
            Undistorted frame with boundary overlay if applicable
        """
        if self._map1 is None or self._map2 is None:
            logger.warning(f"Cannot undistort frame for port {self._camera.port}: not ready")
            return frame

        result = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

        if self._content_expands and self._boundary_rect is not None:
            _draw_dashed_rect(
                result,
                self._boundary_rect[0],
                self._boundary_rect[1],
                self.BOUNDARY_COLOR,
                self.BOUNDARY_THICKNESS,
            )

        return result
