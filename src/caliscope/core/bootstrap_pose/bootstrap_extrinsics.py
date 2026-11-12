"""Initial camera extrinsics from 2D observations, before bundle adjustment.

The dispatcher chooses a method from the state of the points. Each method
returns a deep copy of the cameras with initial, unoptimized extrinsics at
arbitrary scale. The geometry lives in the builder modules beside this one.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.epipolar_pose_builder import build_epipolar_pose_network
from caliscope.core.bootstrap_pose.pose_network_builder import build_pnp_camera_pose_network
from caliscope.core.point_data import ImagePoints, _linked_rows_only
from caliscope.exceptions import CalibrationError

logger = logging.getLogger(__name__)


def bootstrap_extrinsics(points: ImagePoints, cameras: CameraArray) -> CameraArray:
    """Initial extrinsics, choosing the method from the state of the points.

    Unlinked rows (negative object_id) are dropped with a CalibrationWarning
    that counts them. Points with any object geometry (obj_loc) then use
    bootstrap_extrinsics_from_object_geometry, and points without it use
    bootstrap_extrinsics_from_linked_points.

    Raises:
        CalibrationError: if every row is unlinked, points name cameras absent
            from cameras, or a camera lacks intrinsics.
    """
    points = _linked_rows_only(points, "bootstrap_extrinsics")
    if points.df[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].isna().all().all():
        logger.info("No object geometry (obj_loc all NaN); using epipolar bootstrap.")
        return bootstrap_extrinsics_from_linked_points(points, cameras)
    return bootstrap_extrinsics_from_object_geometry(points, cameras)


def bootstrap_extrinsics_from_object_geometry(points: ImagePoints, cameras: CameraArray) -> CameraArray:
    """PnP against known object geometry (obj_loc), then the stereo-pair graph.

    Unlinked rows are dropped with a CalibrationWarning. Cameras the graph
    cannot connect to the largest group stay unposed.

    Raises:
        CalibrationError: if points name cameras absent from cameras, or a
            camera lacks intrinsics.
    """
    points = _linked_rows_only(points, "bootstrap_extrinsics_from_object_geometry")
    _validate_cameras(points, cameras)
    posed = deepcopy(cameras)
    build_pnp_camera_pose_network(points, posed).apply_to(posed)
    return posed


def bootstrap_extrinsics_from_linked_points(points: ImagePoints, cameras: CameraArray) -> CameraArray:
    """Essential matrices on linked 2D correspondences, with third-view resection.

    Ignores obj_loc, so it suits body keypoints. Needs calibrated intrinsics,
    since nothing absorbs focal error. Unlinked rows are dropped with a
    CalibrationWarning. Cameras that cannot be placed stay unposed.

    Raises:
        CalibrationError: if points name cameras absent from cameras, a camera
            lacks intrinsics, fewer than two cameras have observations, or no
            camera pair shares enough correspondences for an essential matrix.
    """
    points = _linked_rows_only(points, "bootstrap_extrinsics_from_linked_points")
    _validate_cameras(points, cameras)
    posed = deepcopy(cameras)
    build_epipolar_pose_network(points, posed).apply_to(posed)
    return posed


def _validate_cameras(points: ImagePoints, cameras: CameraArray) -> None:
    point_cam_ids = set(points.df["cam_id"].unique())
    array_cam_ids = set(cameras.cameras.keys())
    missing_cameras = point_cam_ids - array_cam_ids
    if missing_cameras:
        raise CalibrationError(f"ImagePoints reference cameras {missing_cameras} not in the CameraArray.")

    uncalibrated = [cam_id for cam_id, cam in cameras.cameras.items() if cam.matrix is None or cam.distortions is None]
    if uncalibrated:
        raise CalibrationError(
            f"Cannot run extrinsic calibration -- cameras {uncalibrated} have "
            f"no intrinsic calibration.\n\n"
            f"Run calibrate_intrinsics() for each camera first:\n"
            f"    output = calibrate_intrinsics(points, cameras[{uncalibrated[0]}])\n"
            f"    cameras[{uncalibrated[0]}] = output.camera"
        )
