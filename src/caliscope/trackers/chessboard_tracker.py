"""
Chessboard corner tracker for intrinsic calibration.

Uses OpenCV's findChessboardCorners with sub-pixel refinement.
Unlike CharucoTracker, there is no mirror search — chessboard patterns
don't need it for intrinsic calibration (frames without detection are skipped).
"""

import logging

import cv2
import numpy as np

from caliscope.core.chessboard import Chessboard
from caliscope.packets import PointPacket
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class ChessboardTracker(Tracker):
    """
    Tracker for chessboard calibration patterns.

    Detection is all-or-nothing: either all internal corners are found,
    or none are. This differs from Charuco where partial detection is valid.

    Green visualization color distinguishes from Charuco (red/blue).
    """

    def __init__(self, chessboard: Chessboard) -> None:
        """
        Args:
            chessboard: Chessboard pattern definition (frozen dataclass)
        """
        self.chessboard = chessboard

        # OpenCV findChessboardCorners expects (columns, rows) tuple
        self._pattern_size = (chessboard.columns, chessboard.rows)

        # Detection flags for robustness
        self._flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE

        # Sub-pixel refinement parameters
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,  # max iterations
            0.001,  # epsilon (corner movement threshold)
        )
        self._win_size = (11, 11)  # Search window for sub-pixel refinement

    @property
    def name(self) -> str:
        """Return tracker name for file naming."""
        return "CHESSBOARD"

    def get_points(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        """
        Detect chessboard corners in frame.

        Detection is all-or-nothing: either all (rows * columns) corners
        are found, or an empty PointPacket is returned.

        Args:
            frame: BGR image from video capture
            cam_id: Camera cam_id identifier (unused, for ABC compliance)
            rotation_count: Image rotation in 90-degree increments (unused)

        Returns:
            PointPacket with:
            - point_id: 0 to (rows*columns - 1) in row-major order
            - img_loc: Sub-pixel refined corner positions
            - obj_loc: 3D positions from chessboard.get_object_points()

            Empty PointPacket if detection fails.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Attempt corner detection
        found, corners = cv2.findChessboardCorners(gray, self._pattern_size, flags=self._flags)

        if not found or corners is None:
            # Return empty PointPacket on detection failure
            return PointPacket(
                point_id=np.array([], dtype=np.int32),
                img_loc=np.empty((0, 2), dtype=np.float32),
                obj_loc=np.empty((0, 3), dtype=np.float32),
            )

        # Sub-pixel corner refinement
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            self._win_size,
            (-1, -1),  # zero zone (no dead zone)
            self._criteria,
        )

        # corners_refined shape is (N, 1, 2), flatten to (N, 2)
        img_loc = corners_refined.reshape(-1, 2)

        # Generate point IDs: 0 to N-1 in row-major order
        n_corners = self.chessboard.rows * self.chessboard.columns
        point_id = np.arange(n_corners, dtype=np.int32)

        # Object locations from chessboard geometry
        obj_loc = self.chessboard.get_object_points()

        return PointPacket(point_id=point_id, img_loc=img_loc, obj_loc=obj_loc)

    def get_connected_points(self) -> set[tuple[int, int]]:
        """Point ID pairs forming the grid pattern (adjacent corners only)."""
        return self.chessboard.get_connected_points()

    def get_point_name(self, point_id: int) -> str:
        """Return point ID as string (corners don't have semantic names)."""
        return str(point_id)

    def scatter_draw_instructions(self, point_id: int) -> dict:
        """
        Return drawing parameters for corner visualization.

        Green color (0, 220, 0) in BGR to distinguish from:
        - CharucoTracker: red/blue (0, 0, 220)
        - ArucoTracker: bright green (0, 255, 0)
        """
        return {
            "radius": 5,
            "color": (0, 220, 0),  # Green in BGR
            "thickness": 3,
        }
