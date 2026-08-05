# There may be a mixed functionality here...I'm not sure. Between the corner
# detector and the corner drawer...like, there will need to be something that
# accumulates a frame of corners to be drawn onto the displayed frame.

import logging

import cv2
import numpy as np

from caliscope.packets import PixelFormat, PointPacket
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)

_MIN_SUBPIX_HALF_WIDTH = 2
_MAX_SUBPIX_HALF_WIDTH = 11
_SUBPIX_PITCH_FRACTION = 0.25
_MAX_REFINEMENT_RMS_RATIO = 1.5
_MAX_REFINEMENT_RMS_INCREASE_PX = 0.15
_MIN_REFINED_PITCH_RATIO = 0.5


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

        # Subpixel refinement is sized per frame from the detected corner pitch.
        # A fixed window can include neighboring corners when a board is small in
        # the image and pull distinct observations onto the same feature.
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        self._connected_points = charuco.get_connected_points()

        # Per-camera mirror hint: remembers whether the last successful detection
        # for each camera was on a mirrored image. Avoids the ~107ms penalty of
        # trying the wrong orientation first. Keyed by cam_id — thread-safe because
        # parallel processing guarantees distinct cam_ids per thread.
        self._last_mirrored: dict[int, bool] = {}

    @property
    def name(self):
        return "CHARUCO"

    @property
    def pixel_format(self) -> PixelFormat:
        return PixelFormat.GRAY

    def _detect(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        gray = frame
        if self.charuco.inverted:
            gray = cv2.bitwise_not(gray)

        # Try the orientation that worked last time for this camera
        hint_mirrored = self._last_mirrored.get(cam_id, False)
        try_order = [hint_mirrored, not hint_mirrored]

        ids = np.array([], dtype=np.int32)
        img_loc = np.empty((0, 2), dtype=np.float64)

        detected_mirrored = False
        for is_mirrored in try_order:
            gray_input = cv2.flip(gray, 1) if is_mirrored else gray
            ids, img_loc = self.find_corners_single_frame(gray_input, mirror=is_mirrored)
            if len(ids) > 0:
                self._last_mirrored[cam_id] = is_mirrored
                detected_mirrored = is_mirrored
                break

        # Identity split for a two-sided board with substrate thickness: the
        # back face is a distinct object (object_id=1) whose corners sit at
        # z=+t, directly behind their front counterparts (verified by the
        # correspondence test in tests/test_charuco_tracker.py). At zero
        # thickness both faces share identity so BA fuses them into the same
        # world points — the strongest coupling, and the historical behavior.
        is_back_face = detected_mirrored and self.charuco.thickness_m > 0
        obj_loc = self.get_obj_loc(ids, back_face=is_back_face)
        object_id_value = 1 if is_back_face else 0
        return PointPacket(
            object_id=np.full(len(ids), object_id_value, dtype=np.int32),
            keypoint_id=ids,
            img_loc=img_loc,
            obj_loc=obj_loc,
        )

    def get_point_name(self, keypoint_id: int) -> str:
        return str(keypoint_id)

    def get_connected_points(self) -> set[tuple[int, int]]:
        return self.charuco.get_connected_points()

    def find_corners_single_frame(self, gray_frame, mirror):
        ids = np.array([], dtype=np.int32)
        img_loc = np.empty((0, 2), dtype=np.float64)

        # detectBoard combines marker detection + charuco corner interpolation
        _img_loc, _ids, marker_corners, marker_ids = self.detector.detectBoard(gray_frame)

        if _ids is not None and len(_ids) > 0:
            ids = np.asarray(_ids, dtype=np.int32).reshape(-1)
            raw_img_loc = np.asarray(_img_loc, dtype=np.float32).reshape(-1, 2)
            img_loc = self._refine_corners_safely(gray_frame, ids, raw_img_loc)

            # flip coordinates if mirrored image fed in
            frame_width = gray_frame.shape[1]
            if mirror:
                img_loc[:, 0] = frame_width - img_loc[:, 0]

        return ids, img_loc

    def _refine_corners_safely(
        self,
        gray_frame: np.ndarray,
        ids: np.ndarray,
        raw_img_loc: np.ndarray,
    ) -> np.ndarray:
        """Refine ChArUco corners without allowing neighbor collapse.

        ``CharucoDetector.detectBoard`` already returns interpolated corner
        locations.  ``cornerSubPix`` is therefore an optional improvement, not
        something worth keeping when it makes the known planar grid less
        self-consistent.  The search window is derived from local grid pitch,
        and the unrefined points are retained whenever refinement degrades the
        planar fit or collapses adjacent corners.
        """
        raw_pitch = self._adjacent_pitch(ids, raw_img_loc)
        if raw_pitch is None:
            return raw_img_loc.astype(np.float64)

        half_width = int(
            np.clip(
                np.floor(raw_pitch * _SUBPIX_PITCH_FRACTION),
                _MIN_SUBPIX_HALF_WIDTH,
                _MAX_SUBPIX_HALF_WIDTH,
            )
        )

        try:
            refined = cv2.cornerSubPix(
                gray_frame,
                raw_img_loc.reshape(-1, 1, 2).copy(),
                (half_width, half_width),
                (-1, -1),
                self.criteria,
            ).reshape(-1, 2)
        except Exception as exc:
            logger.debug("Subpixel detection failed: %s", exc)
            return raw_img_loc.astype(np.float64)

        if not np.isfinite(refined).all():
            logger.debug("Discarding non-finite ChArUco subpixel refinement")
            return raw_img_loc.astype(np.float64)

        refined_pitch = self._adjacent_pitch(ids, refined)
        if refined_pitch is None or refined_pitch < raw_pitch * _MIN_REFINED_PITCH_RATIO:
            logger.debug(
                "Discarding ChArUco refinement after adjacent-corner collapse "
                "(raw pitch %.2fpx, refined %.2fpx)",
                raw_pitch,
                refined_pitch if refined_pitch is not None else float("nan"),
            )
            return raw_img_loc.astype(np.float64)

        raw_rms = self._planar_fit_rms(ids, raw_img_loc)
        refined_rms = self._planar_fit_rms(ids, refined)
        allowed_rms = max(
            raw_rms * _MAX_REFINEMENT_RMS_RATIO,
            raw_rms + _MAX_REFINEMENT_RMS_INCREASE_PX,
        )
        if np.isfinite(raw_rms) and (not np.isfinite(refined_rms) or refined_rms > allowed_rms):
            logger.debug(
                "Discarding ChArUco refinement that worsened planar RMS "
                "(raw %.3fpx, refined %.3fpx)",
                raw_rms,
                refined_rms,
            )
            return raw_img_loc.astype(np.float64)

        return refined.astype(np.float64)

    def _adjacent_pitch(self, ids: np.ndarray, img_loc: np.ndarray) -> float | None:
        """Return a conservative local grid pitch from detected adjacent pairs."""
        point_by_id = {int(keypoint_id): point for keypoint_id, point in zip(ids, img_loc, strict=True)}
        distances = [
            float(np.linalg.norm(point_by_id[a] - point_by_id[b]))
            for a, b in self._connected_points
            if a in point_by_id and b in point_by_id
        ]
        if not distances:
            return None
        return float(np.percentile(distances, 10))

    def _planar_fit_rms(self, ids: np.ndarray, img_loc: np.ndarray) -> float:
        """Measure correspondence consistency without assuming camera intrinsics."""
        if len(ids) < 5:
            return 0.0

        obj_loc = np.asarray(self.board.getChessboardCorners(), dtype=np.float32)[ids, :2]
        homography, _ = cv2.findHomography(obj_loc, img_loc.astype(np.float32), 0)
        if homography is None:
            return float("inf")

        projected = cv2.perspectiveTransform(obj_loc[:, None, :], homography)[:, 0, :]
        return float(np.sqrt(np.mean(np.sum((projected - img_loc) ** 2, axis=1))))

    def get_obj_loc(self, ids: np.ndarray, back_face: bool = False):
        """Objective position of charuco corners in a board frame of reference.

        ids are always front-face corner indices (0..n-1) — back-face
        detections keep the same keypoint ids. back_face stamps z=+thickness:
        in the charuco board frame front cameras look along +Z, so the
        substrate extends toward +Z behind the printed front face.
        """
        if len(ids) > 0:
            corners = np.array(self.board.getChessboardCorners()[ids, :])
            # Ensure 3D coordinates (planar boards may return 2D)
            if corners.shape[1] == 2:
                corners = np.column_stack([corners, np.zeros(len(ids))])
            if back_face:
                corners[:, 2] = self.charuco.thickness_m
            return corners
        else:
            return np.empty((0, 3), dtype=np.float64)

    def scatter_draw_instructions(self, keypoint_id: int) -> dict:
        return {"radius": 5, "color": (0, 0, 220), "thickness": 3}
