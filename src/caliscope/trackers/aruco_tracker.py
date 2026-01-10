"""
ArUco marker tracker for extrinsic calibration.

Design decisions:
- Tracks 4 corners per marker for spatial precision
- Point ID scheme: marker_id * 10 + corner_index (1-4)
- obj_loc is always None (no board reference)
- Default dictionary: cv2.aruco.DICT_4X4_100
- Default inversion: False (True only for legacy test data)
- Mirror search: Attempts detection on flipped image if no markers found
"""

import logging

import cv2
import numpy as np

from caliscope.packets import PointPacket
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)

# Corner index constants for clarity
CORNER_TL = 1  # Top-left
CORNER_TR = 2  # Top-right
CORNER_BR = 3  # Bottom-right
CORNER_BL = 4  # Bottom-left


class ArucoTracker(Tracker):
    """
    Tracker for ArUco markers. Detects markers and returns their corner positions.

    WARNING: The mirror_flag_search parameter should ONLY be used when tracking a
    SINGLE double-sided ArUco marker (an "ArUco flag"). Some markers have rotational
    symmetry that can cause ambiguity. When using this feature:
    1. Use exactly ONE marker ID in your dictionary
    2. Print the marker normally on one side
    3. Print the mirrored version on the back side
    4. Ensure cameras on opposite sides can both see the marker

    Note on inversion: The default is False. The test fixture data uses inverted
    markers due to a historical quirk. In production, physical markers should be
    printed normally and used with inverted=False. We may remove this toggle
    once legacy test data is updated.
    """

    def __init__(self, dictionary=cv2.aruco.DICT_4X4_100, inverted=False, mirror_flag_search=False):
        """
        Args:
            dictionary: OpenCV ArUco dictionary to use for detection
            inverted: Whether to invert the image before detection (for legacy test data)
            mirror_search: If True, adds detections of mirror images; only use if a single "aruco flag" is employed
        """
        self.dictionary = dictionary
        self.inverted = inverted
        self.mirror_flag_search = mirror_flag_search  # use with aruco "flag"

        # Create detector instance
        self.dictionary_object = cv2.aruco.getPredefinedDictionary(dictionary)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary_object)

    @property
    def name(self) -> str:
        """Return tracker name for file naming."""
        return "ARUCO"

    def _detect_markers(self, gray_frame):
        """
        Internal helper to detect markers and format results.
        Returns (point_ids, all_corners) or (None, None) if no markers.
        """
        corners, ids, rejected = self.detector.detectMarkers(gray_frame)

        if ids is not None and len(ids) > 0:
            # Flatten corners: each marker has 4 corners in shape (1, 4, 2)
            # We want shape (n_markers * 4, 2)
            all_corners = np.vstack(corners).reshape(-1, 2)

            # Generate point IDs: marker_id * 10 + corner_index (1-4)
            point_ids = []
            for marker_id in ids.flatten():
                base_id = marker_id * 10
                # Add corner indices 1-4 for each marker
                point_ids.extend([base_id + j for j in range(CORNER_TL, CORNER_BL + 1)])

            point_ids = np.array(point_ids, dtype=np.int32)

            # Validate shapes match
            assert len(point_ids) == len(all_corners), "Point ID count must match corner count"

            logger.debug(f"Detected {len(ids)} markers, {len(point_ids)} corners")
            return point_ids, all_corners

        return None, None

    def get_points(self, frame: np.ndarray, port: int = 0, rotation_count: int = 0) -> PointPacket:
        """
        Detect ArUco markers in frame and return corner points.

        Process:
        Invert if configured (for legacy test data)
        Detect markers with cv2.aruco.ArucoDetector
        If no markers found and mirror_search=True, try mirrored image
        Flatten corners and generate point IDs
        Return PointPacket with img_loc only (obj_loc=None)
        """

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Invert if needed (for legacy test data)
        if self.inverted:
            gray_frame = cv2.bitwise_not(gray_frame)

        # Attempt detection on original orientation
        point_ids, all_corners = self._detect_markers(gray_frame)

        # If no markers found and mirror search enabled, try flipped image
        if point_ids is None and self.mirror_flag_search:
            logger.debug("No markers found in original orientation, trying mirrored image")
            mirrored_frame = cv2.flip(gray_frame, 1)  # Horizontal flip
            point_ids, all_corners = self._detect_markers(mirrored_frame)

            # If markers found in mirror, adjust x-coordinates back to original frame and send packet
            if point_ids is not None and all_corners is not None:
                frame_width = gray_frame.shape[1]
                all_corners[:, 0] = frame_width - all_corners[:, 0]
                logger.debug(f"Detected {len(np.unique(point_ids // 10))} markers in mirrored image")

        if point_ids is not None and all_corners is not None:
            return PointPacket(point_id=point_ids, img_loc=all_corners, obj_loc=None)

        # Return empty PointPacket if no markers detected
        return PointPacket(
            point_id=np.array([], dtype=np.int32), img_loc=np.empty((0, 2), dtype=np.float32), obj_loc=None
        )

    def get_point_name(self, point_id: int) -> str:
        """Minimal implementation: return point ID as string."""
        return str(point_id)

    def scatter_draw_instructions(self, point_id: int) -> dict:
        """
        Return drawing parameters for visualizing ArUco corners.
        Green circles to distinguish from CharucoTracker's blue.
        """
        return {
            "radius": 4,
            "color": (0, 255, 0),  # Green in BGR
            "thickness": -1,  # filled circle
        }
