from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

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
    constraint_groups_a: NDArray[np.int32] | None = None,
    constraint_groups_b: NDArray[np.int32] | None = None,
    constraint_distances: NDArray[np.float64] | None = None,
    constraint_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Pixel-space residuals for scipy least_squares, scaled by 1/fx_initial.

    Each distance constraint endpoint is a width-4 group of world-point row
    indices whose mean is the constrained point: a corner endpoint repeats one
    row four times (mean is exactly that row), a centroid endpoint names a
    marker's four corner rows. One code path serves both kinds.
    """
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

    if constraint_groups_a is not None:
        endpoints_a = points_3d[constraint_groups_a].mean(axis=1)
        endpoints_b = points_3d[constraint_groups_b].mean(axis=1)
        diffs = endpoints_a - endpoints_b
        constraint_residuals = (np.linalg.norm(diffs, axis=1) - constraint_distances) * constraint_weights
        return np.concatenate([reproj, constraint_residuals])

    return reproj


# cv2.projectPoints jacobian columns: [rvec 0:3, tvec 3:6, fx 6, fy 7, cx 8, cy 9, dist 10:].
# cv2.fisheye.projectPoints jacobian columns: [fx 0, fy 1, cx 2, cy 3, k 4:8, rvec 8:11, tvec 11:14, alpha 14].
_PINHOLE_RVEC, _PINHOLE_TVEC = slice(0, 3), slice(3, 6)
_FISHEYE_RVEC, _FISHEYE_TVEC = slice(8, 11), slice(11, 14)


def joint_jacobian(
    params: NDArray[np.float64],
    parameterization: BundleParameterization,
    camera_indices: CameraIndices,
    image_coords: ImageCoords,
    obj_indices: NDArray[np.int32],
    constraint_groups_a: NDArray[np.int32] | None = None,
    constraint_groups_b: NDArray[np.int32] | None = None,
    constraint_distances: NDArray[np.float64] | None = None,
    constraint_weights: NDArray[np.float64] | None = None,
) -> csr_matrix:
    """Analytic Jacobian of joint_residuals, as a sparse matrix.

    Same signature as joint_residuals so scipy passes it the same args
    (image_coords and constraint_distances are additive constants in the
    residual, so they don't enter the derivative).

    Camera and intrinsic columns come from the projection Jacobians that
    cv2.projectPoints / cv2.fisheye.projectPoints return. World-point columns
    follow by chain rule: the projection sees the point only through
    x_cam = R @ X + t, so d(proj)/dX = d(proj)/d(tvec) @ R.
    """
    points_3d = params[parameterization.n_camera_params :].reshape(-1, 3)
    world_coords = points_3d[obj_indices]

    n_obs = len(camera_indices)
    n_constraints = len(constraint_groups_a) if constraint_groups_a is not None else 0
    n_residuals = 2 * n_obs + n_constraints
    n_params = parameterization.n_camera_params + 3 * parameterization.n_points

    rows_parts: list[NDArray] = []
    cols_parts: list[NDArray] = []
    data_parts: list[NDArray] = []
    obs_idx = np.arange(n_obs)

    for i, block in enumerate(parameterization.blocks):
        cam_mask = camera_indices == i
        if not cam_mask.any():
            continue

        rvec, tvec, K, dist = parameterization.trial_projection_inputs(params, i)
        cam_world = world_coords[cam_mask].reshape(-1, 1, 3).astype(np.float64)

        if block.fisheye:
            _, jac = cv2.fisheye.projectPoints(cam_world, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist.reshape(4, 1))
            jac_rvec, jac_tvec = jac[:, _FISHEYE_RVEC], jac[:, _FISHEYE_TVEC]
        else:
            _, jac = cv2.projectPoints(cam_world, rvec, tvec, K, dist)
            jac_rvec, jac_tvec = jac[:, _PINHOLE_RVEC], jac[:, _PINHOLE_TVEC]

        cam_columns = [jac_rvec, jac_tvec]
        if block.free_intrinsics:
            # Free params are [s, k1, k2] with fx = s * fx_initial, fy = s * fy_initial,
            # so d/ds = fx_initial * d/dfx + fy_initial * d/dfy.
            jac_s = jac[:, 6:7] * block.fx_initial + jac[:, 7:8] * block.fy_initial
            cam_columns.extend([jac_s, jac[:, 10:12]])
        jac_cam = np.hstack(cam_columns) / block.fx_initial

        rotation = cv2.Rodrigues(rvec)[0]
        jac_points = (jac_tvec @ rotation) / block.fx_initial

        obs_for_cam = obs_idx[cam_mask]
        # Interleaved x/y residual rows, matching projectPoints jacobian row order
        row_map = np.empty(2 * len(obs_for_cam), dtype=np.int64)
        row_map[0::2] = 2 * obs_for_cam
        row_map[1::2] = 2 * obs_for_cam + 1

        off = parameterization.camera_param_offsets[i]
        n_block = block.n_params
        rows_parts.append(np.repeat(row_map, n_block))
        cols_parts.append(np.tile(np.arange(off, off + n_block), len(row_map)))
        data_parts.append(jac_cam.ravel())

        point_cols = parameterization.n_camera_params + 3 * obj_indices[cam_mask].astype(np.int64)
        point_col_triples = np.repeat(point_cols[:, None] + np.arange(3), 2, axis=0)
        rows_parts.append(np.repeat(row_map, 3))
        cols_parts.append(point_col_triples.ravel())
        data_parts.append(jac_points.ravel())

    if constraint_groups_a is not None and n_constraints > 0:
        assert constraint_groups_b is not None and constraint_weights is not None
        endpoints_a = points_3d[constraint_groups_a].mean(axis=1)
        endpoints_b = points_3d[constraint_groups_b].mean(axis=1)
        diffs = endpoints_a - endpoints_b
        norms = np.linalg.norm(diffs, axis=1)
        # Zero subgradient at coincident endpoints, where the norm is non-differentiable
        unit = diffs / np.where(norms > 0, norms, 1.0)[:, None]

        constraint_rows = 2 * n_obs + np.arange(n_constraints, dtype=np.int64)
        for groups, sign in ((constraint_groups_a, 1.0), (constraint_groups_b, -1.0)):
            # 1/4 per group column: a corner endpoint repeats one row 4x and the
            # duplicate COO entries sum back to the full unit vector on that row.
            contribution = (sign * 0.25) * constraint_weights[:, None] * unit
            for col in range(groups.shape[1]):
                group_point_cols = parameterization.n_camera_params + 3 * groups[:, col].astype(np.int64)
                for coord in range(3):
                    rows_parts.append(constraint_rows)
                    cols_parts.append(group_point_cols + coord)
                    data_parts.append(contribution[:, coord])

    # Duplicate (row, col) entries sum during CSR conversion — this is what folds
    # a corner endpoint's four 1/4-contributions back onto its single row.
    coo = coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(rows_parts), np.concatenate(cols_parts))),
        shape=(n_residuals, n_params),
    )
    return csr_matrix(coo)
