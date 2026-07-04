from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bundle_parameterization import BundleParameterization

# Type aliases for clarity
CameraIndices = NDArray[np.int16]  # Shape: (n_observations,)
ImageCoords = NDArray[np.float64]  # Shape: (n_observations, 2)
WorldCoords = NDArray[np.float64]  # Shape: (n_observations, 3) or (n_points, 3)
ErrorsXY = NDArray[np.float64]  # Shape: (n_observations, 2)


def project_points(world: NDArray, rvec: NDArray, tvec: NDArray, K: NDArray, dist: NDArray, fisheye: bool) -> NDArray:
    """Project 3D points to 2D pixel coordinates.

    Dispatches between Brown-Conrady and fisheye equidistant models.
    """
    pts = world.reshape(-1, 1, 3).astype(np.float64)
    if fisheye:
        d = dist.ravel()
        if d.shape[0] != 4:
            raise ValueError(f"Fisheye projection requires 4 distortion coefficients, got {d.shape[0]}")
        projected, _ = cv2.fisheye.projectPoints(pts, rvec.reshape(3, 1), tvec.reshape(3, 1), K, d.reshape(4, 1))
        return projected.reshape(-1, 2)
    else:
        projected, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
        return projected.reshape(-1, 2)


def reprojection_errors(
    camera_array: CameraArray,
    camera_indices: CameraIndices,
    image_coords: ImageCoords,
    world_coords: WorldCoords,
) -> ErrorsXY:
    """Pixel-space reprojection errors for reporting.

    Uses each camera's stored intrinsics and extrinsics. Dispatches fisheye
    cameras to the equidistant model automatically.
    """
    errors_xy = np.zeros_like(image_coords)

    for cam_id, camera_data in camera_array.posed_cameras.items():
        camera_index = camera_array.posed_cam_id_to_index[cam_id]
        cam_mask = camera_indices == camera_index

        if not cam_mask.any():
            continue

        cam_world_coords = world_coords[cam_mask]
        cam_observed = image_coords[cam_mask]

        if camera_data.matrix is None or camera_data.distortions is None:
            raise ValueError(f"Camera {cam_id} missing intrinsics for pixel-mode reprojection")
        if camera_data.rotation is None or camera_data.translation is None:
            raise ValueError(f"Camera {cam_id} missing extrinsics")

        rvec, _ = cv2.Rodrigues(camera_data.rotation)
        rvec = rvec.ravel()
        tvec = camera_data.translation

        projected = project_points(
            cam_world_coords, rvec, tvec, camera_data.matrix, camera_data.distortions, camera_data.fisheye
        )
        errors_xy[cam_mask] = projected - cam_observed

    return errors_xy


def joint_residuals(
    params: NDArray[np.float64],
    parameterization: BundleParameterization,
    camera_indices: CameraIndices,
    image_coords: ImageCoords,
    obj_indices: NDArray[np.int32],
    constraint_pairs: NDArray[np.int32] | None = None,
    constraint_distances: NDArray[np.float64] | None = None,
    constraint_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Pixel-space residuals for scipy least_squares, scaled by 1/fx_initial."""
    points_3d = params[parameterization.n_camera_params :].reshape(-1, 3)
    world_coords = points_3d[obj_indices]

    errors_xy = np.zeros_like(image_coords)

    for i, block in enumerate(parameterization.blocks):
        cam_mask = camera_indices == i
        if not cam_mask.any():
            continue

        rvec, tvec, K, dist = parameterization.trial_projection_inputs(params, i)
        cam_world = world_coords[cam_mask]
        cam_observed = image_coords[cam_mask]

        projected = project_points(cam_world, rvec, tvec, K, dist, block.fisheye)
        errors_xy[cam_mask] = (projected - cam_observed) / block.fx_initial

    reproj = errors_xy.ravel()

    if constraint_pairs is not None:
        diffs = points_3d[constraint_pairs[:, 0]] - points_3d[constraint_pairs[:, 1]]
        constraint_residuals = (np.linalg.norm(diffs, axis=1) - constraint_distances) * constraint_weights
        return np.concatenate([reproj, constraint_residuals])

    return reproj
