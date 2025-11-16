"""
ArUco marker tracker for extrinsic calibration.

Design decisions:
- Tracks 4 corners per marker for spatial precision
- Point ID scheme: marker_id * 10 + corner_index (1-4)
- obj_loc is always None (no board reference)
- Default dictionary: cv2.aruco.DICT_4X4_100
- Default inversion: False (True only for legacy test data)
"""

import logging

import cv2
import numpy as np

from caliscope.packets import PointPacket
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class ArucoTracker(Tracker):
    """
    Tracker for ArUco markers. Detects markers and returns their corner positions.

    Note on inversion: The default is False. The test fixture data uses inverted
    markers due to a historical quirk. In production, physical markers should be
    printed normally and used with inverted=False. We may remove this toggle
    once legacy test data is updated.
    """

    def __init__(self, dictionary=cv2.aruco.DICT_4X4_100, inverted=False):
        """
        Args:
            dictionary: OpenCV ArUco dictionary to use for detection
            inverted: Whether to invert the image before detection (for legacy test data)
        """
        self.dictionary = dictionary
        self.inverted = inverted

        # Create detector instance
        self.dictionary_object = cv2.aruco.getPredefinedDictionary(dictionary)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary_object)

    @property
    def name(self) -> str:
        """Return tracker name for file naming."""
        return "ARUCO"

    def get_points(self, frame: np.ndarray, port: int, rotation_count: int) -> PointPacket:
        """
        Detect ArUco markers in frame and return corner points.

        Process:
        1. Apply rotation if needed (using rotation_count)
        2. Convert to grayscale
        3. Invert if configured (for legacy test data)
        4. Detect markers with cv2.aruco.ArucoDetector
        5. Flatten corners and generate point IDs
        6. Return PointPacket with img_loc only (obj_loc=None)
        """
        # Apply rotation if needed
        from caliscope.trackers.helper import apply_rotation

        rotated_frame = apply_rotation(frame, rotation_count)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

        # Invert if needed (for legacy test data)
        if self.inverted:
            gray_frame = cv2.bitwise_not(gray_frame)

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray_frame)

        # Process detections into PointPacket format
        if ids is not None and len(ids) > 0:
            # Flatten corners: each marker has 4 corners in shape (1, 4, 2)
            # We want shape (n_markers * 4, 2)
            all_corners = np.vstack(corners).reshape(-1, 2)

            # Generate point IDs: marker_id * 10 + corner_index (1-4)
            point_ids = []
            for i, marker_id in enumerate(ids.flatten()):
                base_id = marker_id * 10
                # Add corner indices 1-4 for each marker
                point_ids.extend([base_id + j for j in range(1, 5)])

            point_ids = np.array(point_ids, dtype=np.int32)

            # Validate shapes match
            assert len(point_ids) == len(all_corners), "Point ID count must match corner count"

            logger.debug(f"Detected {len(ids)} markers, {len(point_ids)} corners")

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
        Green circles
        """
        return {
            "radius": 4,
            "color": (0, 255, 0),  # Green in BGR
            "thickness": -1,  # filled circle
        }
