# caliscope/calibration/capture_volume/helper_functions/get_point_estimates.py

import logging

from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.point_data import ImagePoints, WorldPoints

logger = logging.getLogger(__name__)


def create_point_estimates_from_stereopairs(camera_array: CameraArray, image_points: ImagePoints) -> PointEstimates:
    """
    Create PointEstimates from stereo triangulated points.
    Replaces legacy pipeline with direct conversion from ImagePoints.triangulate().
    """
    logger.info("Creating point estimates using ImagePoints.triangulate()")

    # Use the clean triangulation implementation
    world_points: WorldPoints = image_points.triangulate(camera_array)

    return world_points.to_point_estimates()
