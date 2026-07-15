from __future__ import annotations

import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.epipolar_pose_builder import build_epipolar_pose_network
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.core.bootstrap_pose.pose_network_builder import PoseNetworkBuilder
from caliscope.core.point_data import ImagePoints

logger = logging.getLogger(__name__)


def build_paired_pose_network(
    image_points: ImagePoints,
    camera_array: CameraArray,
) -> PairedPoseNetwork:
    """Build the stereo-pair pose graph, selecting the strategy from the data.

    With known object geometry (obj_loc populated) the PnP path resections each
    camera against the calibration target. With 2D-only observations (obj_loc all
    NaN -- e.g. body keypoints) the essential-matrix path recovers relative poses
    from 2D-2D correspondences. Both return a PairedPoseNetwork.
    """
    obj_cols = image_points.df[["obj_loc_x", "obj_loc_y", "obj_loc_z"]]
    if obj_cols.isna().all().all():
        logger.info("No object geometry (obj_loc all NaN); using epipolar bootstrap.")
        return build_epipolar_pose_network(image_points, camera_array)

    builder = PoseNetworkBuilder(camera_array, image_points)
    return builder.estimate_camera_to_object_poses().estimate_relative_poses().filter_outliers(threshold=1.5).build()
