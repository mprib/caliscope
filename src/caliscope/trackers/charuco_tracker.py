# There may be a mixed functionality here...I'm not sure. Between the corner
# detector and the corner drawer...like, there will need to be something that
# accumulates a frame of corners to be drawn onto the displayed frame.

import logging

import cv2
import numpy as np

from caliscope.packets import PointPacket
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class CharucoTracker(Tracker):
    def __init__(self, charuco):
        # need camera to know resolution and to assign calibration parameters
        # to camera
        self.charuco = charuco
        self.board = charuco.board

        # Widen the adaptive threshold step to reduce passes from 3 to 2.
        # Default step=10 with range [3,23] yields windows {3,13,23}.
        # Step=20 yields {3,23} — skips the middle pass for ~2x faster failure paths
        # with negligible detection loss on well-lit calibration boards.
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeStep = 20

        self.detector = cv2.aruco.CharucoDetector(self.board, detectorParams=params)

        # for subpixel corner correction
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        self.conv_size = (11, 11)  # Don't make this too large.

        # Per-camera mirror hint: remembers whether the last successful detection
        # for each camera was on a mirrored image. Avoids the ~107ms penalty of
        # trying the wrong orientation first. Keyed by cam_id — thread-safe because
        # parallel processing guarantees distinct cam_ids per thread.
        self._last_mirrored: dict[int, bool] = {}

    @property
    def name(self):
        return "CHARUCO"

    def get_points(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        """Detect charuco corners, trying the last-known orientation first.

        Uses a mirror hint per camera to avoid the ~107ms penalty of attempting
        detection in the wrong orientation. Falls back to the other orientation
        only if the hinted one fails.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.charuco.inverted:
            gray = cv2.bitwise_not(gray)

        # Try the orientation that worked last time for this camera
        hint_mirrored = self._last_mirrored.get(cam_id, False)
        try_order = [hint_mirrored, not hint_mirrored]

        ids = np.array([], dtype=np.int32)
        img_loc = np.empty((0, 2), dtype=np.float64)

        for is_mirrored in try_order:
            gray_input = cv2.flip(gray, 1) if is_mirrored else gray
            ids, img_loc = self.find_corners_single_frame(gray_input, mirror=is_mirrored)
            if ids.any():
                self._last_mirrored[cam_id] = is_mirrored
                break

        obj_loc = self.get_obj_loc(ids)
        return PointPacket(ids, img_loc, obj_loc)

    def get_point_name(self, point_id: int) -> str:
        return str(point_id)

    def get_connected_points(self) -> set[tuple[int, int]]:
        return self.charuco.get_connected_points()

    def find_corners_single_frame(self, gray_frame, mirror):
        ids = np.array([], dtype=np.int32)
        img_loc = np.empty((0, 2), dtype=np.float64)

        # detectBoard combines marker detection + charuco corner interpolation
        _img_loc, _ids, marker_corners, marker_ids = self.detector.detectBoard(gray_frame)

        if _ids is not None and len(_ids) > 0:
            # Sub-pixel refinement — occasionally errors out, so just move along if it fails
            try:
                _img_loc = cv2.cornerSubPix(
                    gray_frame,
                    _img_loc,
                    self.conv_size,
                    (-1, -1),
                    self.criteria,
                )
            except Exception as e:
                logger.debug(f"Sub pixel detection failed: {e}")

            ids = _ids[:, 0]
            img_loc = _img_loc[:, 0]

            # flip coordinates if mirrored image fed in
            frame_width = gray_frame.shape[1]
            if mirror:
                img_loc[:, 0] = frame_width - img_loc[:, 0]

        return ids, img_loc

    def get_obj_loc(self, ids: np.ndarray):
        """Objective position of charuco corners in a board frame of reference"""
        if len(ids) > 0:
            corners = self.board.getChessboardCorners()[ids, :]
            # Ensure 3D coordinates (planar boards may return 2D)
            if corners.shape[1] == 2:
                corners = np.column_stack([corners, np.zeros(len(ids))])
            return corners
        else:
            return np.empty((0, 3), dtype=np.float64)

    # @property
    def scatter_draw_instructions(self, point_id: int) -> dict:
        rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        return rules
