"""
Chessboard corner tracker.

Uses OpenCV's findChessboardCorners with sub-pixel refinement. Serves
intrinsic calibration always; when its Chessboard carries a square_size_cm the
emitted obj_loc is metric, so the same tracker also drives extrinsic
calibration (the Calibration Target Interchangeability contract).

Unlike CharucoTracker, there is no mirror search — chessboard patterns
don't need it for intrinsic calibration (frames without detection are skipped).
"""

import logging

import cv2
import numpy as np

from caliscope.core.chessboard import Chessboard
from caliscope.packets import PixelFormat, PointPacket
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


def _subpix_window_half_width(corners_row_major: np.ndarray, columns: int, rows: int) -> int:
    """Sub-pixel search-window half-width derived from detected corner pitch.

    corners_row_major is the raw findChessboardCorners output (shape (N, 1, 2)
    or (N, 2)), row-major in (columns, rows) order. A window wider than ~a
    quarter of the corner pitch drags corners toward neighbors: on OpenCap Cam0
    (16 px squares) the fixed 11 px window inflated the planar-homography
    residual to 4-8 px vs 0.12 px at a 5 px window. The pitch is the min over
    both horizontal and vertical neighbors — perspective foreshortening (camera
    well off the board plane) can make the smallest pitch vertical. Half-width =
    clamp(floor(min_neighbor_px / 4), 2, 11); 11 stays the large-board ceiling
    so GUI-scale boards are unchanged.
    """
    grid = corners_row_major.reshape(rows, columns, 2)
    horizontal_px = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical_px = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    min_neighbor_px = float(min(horizontal_px.min(), vertical_px.min()))
    return int(np.clip(np.floor(min_neighbor_px / 4), 2, 11))


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
        # Sub-pixel search window is derived per frame from the detected
        # corner pitch (see _subpix_window_half_width), not fixed.

    @property
    def name(self) -> str:
        """Return tracker name for file naming."""
        return "CHESSBOARD"

    @property
    def pixel_format(self) -> PixelFormat:
        return PixelFormat.GRAY

    def _detect(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        gray = frame

        # Attempt corner detection
        found, corners = cv2.findChessboardCorners(gray, self._pattern_size, flags=self._flags)

        if not found or corners is None:
            # Return empty PointPacket on detection failure
            return PointPacket(
                object_id=np.array([], dtype=np.int32),
                keypoint_id=np.array([], dtype=np.int32),
                img_loc=np.empty((0, 2), dtype=np.float32),
                obj_loc=np.empty((0, 3), dtype=np.float32),
            )

        # Sub-pixel corner refinement, window scaled to the board's frame size
        half_width = _subpix_window_half_width(corners, self.chessboard.columns, self.chessboard.rows)
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (half_width, half_width),
            (-1, -1),  # zero zone (no dead zone)
            self._criteria,
        )

        # corners_refined shape is (N, 1, 2), flatten to (N, 2)
        img_loc = corners_refined.reshape(-1, 2)

        # Generate point IDs: 0 to N-1 in row-major order
        n_corners = self.chessboard.rows * self.chessboard.columns
        keypoint_id = np.arange(n_corners, dtype=np.int32)
        object_id = np.zeros(n_corners, dtype=np.int32)

        # Object locations from chessboard geometry
        obj_loc = self.chessboard.get_object_points()

        return PointPacket(object_id=object_id, keypoint_id=keypoint_id, img_loc=img_loc, obj_loc=obj_loc)

    def get_connected_points(self) -> set[tuple[int, int]]:
        """Point ID pairs forming the grid pattern (adjacent corners only)."""
        return self.chessboard.get_connected_points()

    def get_point_name(self, keypoint_id: int) -> str:
        return str(keypoint_id)

    def scatter_draw_instructions(self, keypoint_id: int) -> dict:
        """Green color (0, 220, 0) in BGR to distinguish from:
        - CharucoTracker: red/blue (0, 0, 220)
        - ArucoTracker: bright green (0, 255, 0)
        """
        return {
            "radius": 5,
            "color": (0, 220, 0),  # Green in BGR
            "thickness": 3,
        }
