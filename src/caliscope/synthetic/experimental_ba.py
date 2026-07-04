# TEMPORARY: delete when production pixel-space BA lands (roadmap Phase 3).
# See tasks.json: delete-experimental-ba.
"""Experimental bundle adjustment variants for testbed use.

Zero production code changes. These functions replicate the production
optimize() setup and vary only the solver parameters (loss function,
free intrinsic set). When the roadmap's production implementation lands,
the experiments migrate onto it and this module is deleted.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares

from caliscope.core.capture_volume import CaptureVolume, OptimizationStatus
from caliscope.core.point_data import WorldPoints
from caliscope.core.reprojection import (
    CameraIndices,
    ImageCoords,
    bundle_residuals,
)

logger = logging.getLogger(__name__)

_SCIPY_STATUS_REASONS = {
    -1: "improper_input",
    0: "max_evaluations",
    1: "converged_gtol",
    2: "converged_ftol",
    3: "converged_xtol",
    4: "converged_ftol_and_xtol",
}


def optimize_with_robust_loss(
    capture_volume: CaptureVolume,
    *,
    loss: str = "huber",
    f_scale_px: float = 1.0,
    ftol: float = 1e-8,
    max_nfev: int = 1000,
    verbose: int = 0,
    strict: bool = True,
    use_constraints: bool = True,
    pixel_sigma: float = 1.0,
) -> CaptureVolume:
    """Production-layout BA with a robust loss function.

    Replicates CaptureVolume.optimize() exactly — same residuals, sparsity,
    parameter layout — and changes only the scipy loss and f_scale.

    f_scale_px is the inlier scale in pixels. Production residuals are in
    normalized coordinates, so f_scale_px is divided by the median focal
    length to convert: 1.0 px ≈ 7.2e-4 normalized at the WEBCAM profile.
    """
    matched_mask = capture_volume.img_to_obj_map >= 0
    posed_cam_ids = set(capture_volume.camera_array.posed_cam_id_to_index.keys())
    posed_mask: np.ndarray = capture_volume.image_points.df["cam_id"].isin(posed_cam_ids).to_numpy()
    combined_mask = matched_mask & posed_mask

    matched_img_df = capture_volume.image_points.df[combined_mask]

    camera_indices: CameraIndices = np.array(
        [capture_volume.camera_array.posed_cam_id_to_index[cam_id] for cam_id in matched_img_df["cam_id"]],
        dtype=np.int16,
    )
    image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
    image_to_world_indices = capture_volume.img_to_obj_map[combined_mask]

    initial_params = capture_volume._get_vectorized_params()

    constraint_pairs = None
    constraint_distances = None
    constraint_weights = None

    focal_lengths = [
        cam.matrix[0, 0] for cam in capture_volume.camera_array.posed_cameras.values() if cam.matrix is not None
    ]
    f_median = float(np.median(focal_lengths))

    if use_constraints and capture_volume.constraints is not None:
        arrays = capture_volume._build_constraint_arrays()
        if arrays is not None:
            constraint_pairs, constraint_distances, constraint_sigmas = arrays
            constraint_weights = (pixel_sigma / f_median) / constraint_sigmas

    sparsity_pattern = capture_volume._get_sparsity_pattern(camera_indices, image_to_world_indices, constraint_pairs)

    # Convert pixel-space f_scale to normalized-coordinate units
    f_scale_normalized = f_scale_px / f_median

    result = least_squares(
        bundle_residuals,
        initial_params,
        args=(
            capture_volume.camera_array,
            camera_indices,
            image_coords,
            image_to_world_indices,
            True,
            constraint_pairs,
            constraint_distances,
            constraint_weights,
        ),
        jac_sparsity=sparsity_pattern,
        verbose=verbose,
        x_scale="jac",
        loss=loss,
        f_scale=f_scale_normalized,
        ftol=ftol,
        max_nfev=max_nfev,
        method="trf",
    )

    termination_reason = _SCIPY_STATUS_REASONS.get(result.status, f"unknown_{result.status}")
    converged = result.status in (1, 2, 3, 4)

    optimization_status = OptimizationStatus(
        converged=converged,
        termination_reason=termination_reason,
        iterations=result.nfev,
        final_cost=float(result.cost),
    )

    if strict and not converged:
        from caliscope.exceptions import CalibrationError

        raise CalibrationError(
            f"Robust BA did not converge: {termination_reason}\n"
            f"Pass strict=False to suppress this error and inspect the result."
        )

    new_camera_array = deepcopy(capture_volume.camera_array)
    new_camera_array.update_extrinsic_params(result.x)

    n_cams = len(capture_volume.camera_array.posed_cameras)
    n_cam_params = 6
    optimized_points = result.x[n_cams * n_cam_params :].reshape((-1, 3))

    new_world_df = capture_volume.world_points.df.copy()
    new_world_df[["x_coord", "y_coord", "z_coord"]] = optimized_points

    return CaptureVolume(
        camera_array=new_camera_array,
        image_points=capture_volume.image_points,
        world_points=WorldPoints(new_world_df),
        constraints=capture_volume.constraints,
        _optimization_status=optimization_status,
    )
