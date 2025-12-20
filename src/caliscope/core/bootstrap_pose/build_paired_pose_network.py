# caliscope/calibration/array_initialization/estimate_pairwise_extrinsics.py
from __future__ import annotations

import logging
from typing import Literal

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.post_processing.point_data import ImagePoints
from caliscope.core.bootstrap_pose.legacy_stereocal_paired_pose_network import (
    build_legacy_stereocal_paired_pose_network,
)
from caliscope.core.bootstrap_pose.pose_network_builder import PoseNetworkBuilder

logger = logging.getLogger(__name__)


def build_paired_pose_network(
    image_points: ImagePoints,
    camera_array: CameraArray,
    method: Literal["pnp", "stereocalibrate"] = "pnp",
) -> PairedPoseNetwork:
    if method == "stereocalibrate":
        network = build_legacy_stereocal_paired_pose_network(image_points, camera_array)
        return network
    else:
        builder = PoseNetworkBuilder(camera_array, image_points)
        network = (
            builder.estimate_camera_to_object_poses(min_points=6)
            .estimate_relative_poses()
            .filter_outliers(threshold=1.5)
            .build()
        )
        return network
