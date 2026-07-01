from __future__ import annotations

import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.core.point_data import ImagePoints
from caliscope.core.bootstrap_pose.pose_network_builder import PoseNetworkBuilder

logger = logging.getLogger(__name__)


def build_paired_pose_network(
    image_points: ImagePoints,
    camera_array: CameraArray,
) -> PairedPoseNetwork:
    builder = PoseNetworkBuilder(camera_array, image_points)
    return builder.estimate_camera_to_object_poses().estimate_relative_poses().filter_outliers(threshold=1.5).build()
