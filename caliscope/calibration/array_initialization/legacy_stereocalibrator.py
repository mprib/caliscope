# caliscope/calibration/array_initialization/legacy_stereocalibrator.py
from __future__ import annotations

import logging

from caliscope.calibration.array_initialization.estimate_pairwise_extrinsics import (
    estimate_pairwise_extrinsics,
)
from caliscope.calibration.array_initialization.stereopair_graph import StereoPairGraph
from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.point_data import ImagePoints

logger = logging.getLogger(__name__)


class LegacyStereoCalibrator:
    """
    Legacy stereo calibrator class.

    Refactored to delegate to the new function-based API internally.
    Maintained temporarily for backward compatibility during transition.

    TODO: Deprecate this class once all call sites are updated to use
    estimate_pairwise_extrinsics() directly.
    """

    def __init__(self, camera_array: CameraArray, image_points: ImagePoints):
        """
        Args:
            camera_array: Camera array with intrinsic parameters
            image_points: 2D point correspondences across cameras
        """
        self.camera_array = camera_array
        self.image_points = image_points

    def stereo_calibrate_all(self, boards_sampled: int = 10) -> StereoPairGraph:
        """
        Estimate pairwise extrinsics for all camera pairs.

        Returns:
            StereoPairGraph containing all successfully estimated stereo pairs

        Note:
            BREAKING CHANGE: Previously returned dict[str, dict].
            Update call sites to use StereoPairGraph.apply_to(camera_array)
            instead of CameraArrayInitializer.
        """
        logger.info("LegacyStereoCalibrator delegating to estimate_pairwise_extrinsics()")

        return estimate_pairwise_extrinsics(
            image_points=self.image_points,
            camera_array=self.camera_array,
            boards_sampled=boards_sampled,
        )
