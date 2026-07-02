import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray

# Type aliases for clarity
CameraIndices = NDArray[np.int16]  # Shape: (n_observations,)
ImageCoords = NDArray[np.float64]  # Shape: (n_observations, 2)
WorldCoords = NDArray[np.float64]  # Shape: (n_observations, 3) or (n_points, 3)
ErrorsXY = NDArray[np.float64]  # Shape: (n_observations, 2)


def reprojection_errors(
    camera_array: CameraArray,
    camera_indices: CameraIndices,  # (n_observations,)
    image_coords: ImageCoords,  # (n_observations, 2)
    world_coords: WorldCoords,  # (n_observations, 3)
    use_normalized: bool = False,
    extrinsics_override: NDArray[np.float64] | None = None,  # (n_cams, 6) rvec+tvec per camera
) -> ErrorsXY:  # Returns: (n_observations, 2)
    """
    Core projection logic. Returns (n_observations, 2) error array.
    This is the ONLY place that calls cv2.projectPoints.

    Two modes for different use cases:
    - normalized: Undistorts observations to normalized plane, projects with identity K.
                  Better numerical conditioning for optimization (see Triggs et al.).
    - pixels: Keeps distorted observations, projects with full camera model.
              Reports error in original image coordinates (intuitive for users).

    Args:
        camera_array: CameraArray with posed cameras (used for intrinsics and cam_id mapping)
        camera_indices: Array mapping each observation to a camera index
        image_coords: Observed 2D image coordinates (distorted pixel coords)
        world_coords: 3D world coordinates (one per observation)
        use_normalized: If True, compute error in normalized coords (for optimization)
                        If False, compute error in distorted pixel coords (for reporting)
        extrinsics_override: If provided, use these extrinsics instead of camera_array's.
                             Shape (n_cams, 6) where each row is [rvec(3), tvec(3)].
                             Used by bundle_residuals to avoid mutating camera_array.

    Returns:
        errors_xy: (n_observations, 2) array of x,y reprojection errors
    """
    errors_xy = np.zeros_like(image_coords)

    for cam_id, camera_data in camera_array.posed_cameras.items():
        camera_index = camera_array.posed_cam_id_to_index[cam_id]
        cam_mask = camera_indices == camera_index

        if not cam_mask.any():
            continue

        # Get data for this camera
        cam_world_coords = world_coords[cam_mask]  # (n_cam_obs, 3)
        cam_observed = image_coords[cam_mask]  # (n_cam_obs, 2)

        # Select coordinate system and camera model
        if use_normalized:
            # Normalized mode: undistort observations, project with identity K
            cam_observed = camera_data.undistort_points(cam_observed, output="normalized")
            cam_matrix = np.identity(3)
            dist_coeffs = np.zeros(5)  # No distortion in normalized space
        else:
            # Pixel mode: keep distorted observations, project with full model
            if camera_data.matrix is None or camera_data.distortions is None:
                raise ValueError(f"Camera {cam_id} missing intrinsics for pixel-mode reprojection")
            cam_matrix = camera_data.matrix
            dist_coeffs = camera_data.distortions

        # Get extrinsics: from override if provided, otherwise from camera_data
        if extrinsics_override is not None:
            rvec = extrinsics_override[camera_index, :3]
            tvec = extrinsics_override[camera_index, 3:6]
        else:
            # posed_cameras guarantees rotation/translation are present
            if camera_data.rotation is None or camera_data.translation is None:
                raise ValueError(f"Camera {cam_id} missing extrinsics")
            rvec, _ = cv2.Rodrigues(camera_data.rotation)
            rvec = rvec.ravel()
            tvec = camera_data.translation

        # Project 3D points to 2D
        projected, _ = cv2.projectPoints(
            cam_world_coords.reshape(-1, 1, 3),
            rvec,
            tvec,
            cam_matrix,
            dist_coeffs,
        )
        projected = projected.reshape(-1, 2)  # (n_cam_obs, 2)
        errors_xy[cam_mask] = projected - cam_observed

    return errors_xy  # (n_observations, 2)


def bundle_residuals(
    params: NDArray[np.float64],  # Shape: (n_camera_params + n_points*3,)
    camera_array: CameraArray,
    camera_indices: CameraIndices,  # (n_observations,)
    image_coords: ImageCoords,  # (n_observations, 2)
    obj_indices: NDArray[np.int32],  # (n_observations,)
    use_normalized: bool = True,
    constraint_pairs: NDArray[np.int32] | None = None,
    constraint_distances: NDArray[np.float64] | None = None,
    constraint_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:  # Returns: (n_observations*2 + n_constraints,)
    """
    Callback for scipy.optimize.least_squares.

    NOTE: This function does NOT mutate camera_array. Extrinsics are extracted
    from the params vector and passed to reprojection_errors via override.
    This is critical because least_squares evaluates at trial values that may
    be rejected - mutating would corrupt the camera_array with rejected trials.

    Constraint rows are appended after reprojection rows. Reprojection rows
    occupy positions 0..n_observations*2-1; any consumer computing reprojection
    RMSE must slice to that range.
    """
    n_cams = len(camera_array.posed_cameras)
    n_cam_params = 6

    extrinsics = params[: n_cams * n_cam_params].reshape((n_cams, n_cam_params))
    points_3d = params[n_cams * n_cam_params :].reshape((-1, 3))
    world_coords = points_3d[obj_indices]

    errors_xy = reprojection_errors(
        camera_array,
        camera_indices,
        image_coords,
        world_coords,
        use_normalized,
        extrinsics_override=extrinsics,
    )
    reproj = errors_xy.ravel()

    if constraint_pairs is not None:
        diffs = points_3d[constraint_pairs[:, 0]] - points_3d[constraint_pairs[:, 1]]
        constraint_residuals = (np.linalg.norm(diffs, axis=1) - constraint_distances) * constraint_weights
        return np.concatenate([reproj, constraint_residuals])

    return reproj
