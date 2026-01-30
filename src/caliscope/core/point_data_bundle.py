from __future__ import annotations
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from copy import deepcopy
from numpy.typing import NDArray

import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal
import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.core.reprojection import (
    ErrorsXY,
    reprojection_errors,
    bundle_residuals,
    ImageCoords,
    WorldCoords,
    CameraIndices,
)
from caliscope.core.reprojection_report import ReprojectionReport
from caliscope.core.alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
    SimilarityTransform,
)
from caliscope.core.scale_accuracy import (
    compute_scale_accuracy as compute_scale_accuracy_impl,
    ScaleAccuracyData,
)

import pandas as pd

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class OptimizationStatus:
    """Result metadata from bundle adjustment optimization.

    Populated by optimize(), cleared by filter methods.
    """

    converged: bool
    termination_reason: str  # "converged_gtol", "max_evaluations", etc.
    iterations: int  # nfev from scipy
    final_cost: float


# Mapping from scipy least_squares status codes to human-readable reasons
_SCIPY_STATUS_REASONS: dict[int, str] = {
    -1: "improper_input",
    0: "max_evaluations",
    1: "converged_gtol",
    2: "converged_ftol",
    3: "converged_xtol",
    4: "converged_small_step",
}


@dataclass(frozen=True)
class PointDataBundle:
    camera_array: CameraArray
    image_points: ImagePoints
    world_points: WorldPoints
    # Computed field: maps each image observation to its world point index (-1 if unmatched)
    img_to_obj_map: np.ndarray = field(init=False)
    # Optimization metadata: None if bundle hasn't been optimized or was filtered post-optimization
    _optimization_status: OptimizationStatus | None = field(default=None, compare=False)

    @property
    def optimization_status(self) -> OptimizationStatus | None:
        """Optimization result metadata, or None if not from optimize() call."""
        return self._optimization_status

    def __post_init__(self):
        """Compute mapping and validate geometry."""
        object.__setattr__(self, "img_to_obj_map", self._compute_img_to_obj_map())
        self._validate_geometry()

    def _validate_geometry(self):
        """Ensure data counts make geometric sense."""
        n_img = len(self.image_points.df)
        n_world = len(self.world_points.df)
        n_cams = len(self.camera_array.posed_cameras)

        if n_img == 0:
            raise ValueError("No image observations provided")
        if n_world == 0:
            raise ValueError("No world points provided")
        if n_cams == 0:
            raise ValueError("No posed cameras in array")

        # Check that we have at least some matched observations
        n_matched = np.sum(self.img_to_obj_map >= 0)
        if n_matched == 0:
            raise ValueError("No image observations have corresponding world points")

        if n_matched < n_world * 2:
            logger.warning(
                f"Suspicious geometry: {n_matched} matched observations for {n_world} world points. "
                f"Expected at least {n_world * 2} for multi-view geometry."
            )
        # Validate indices are in bounds
        valid_indices = self.img_to_obj_map[self.img_to_obj_map >= 0]
        if valid_indices.size > 0 and valid_indices.max() >= n_world:
            raise ValueError(f"obj_indices contains out-of-bounds index: {valid_indices.max()} >= {n_world}")

    def _compute_img_to_obj_map(self) -> np.ndarray:
        """Map each image observation to its world point index. Returns -1 for unmatched."""
        # Same logic as before, just renamed
        world_df = self.world_points.df.reset_index().rename(columns={"index": "world_idx"})
        mapping = world_df.set_index(["sync_index", "point_id"])["world_idx"].to_dict()

        img_df = self.image_points.df
        keys = list(zip(img_df["sync_index"], img_df["point_id"]))
        img_to_obj_map = np.array([mapping.get(key, -1) for key in keys], dtype=np.int32)

        n_unmatched = np.sum(img_to_obj_map == -1)
        if n_unmatched > 0:
            logger.info(f"{n_unmatched} of {len(img_to_obj_map)} image observations have no world point")

        return img_to_obj_map

    @cached_property
    def reprojection_report(self) -> ReprojectionReport:
        """
        Generate comprehensive reprojection error report in pixel units.
        Cached automatically since bundle data is immutable.
        """
        # 1. Filter to matched observations from posed cameras only
        matched_mask = self.img_to_obj_map >= 0
        posed_ports = set(self.camera_array.posed_port_to_index.keys())
        posed_mask: np.ndarray = self.image_points.df["port"].isin(posed_ports).to_numpy()
        combined_mask = matched_mask & posed_mask

        n_total = len(self.img_to_obj_map)
        n_matched = combined_mask.sum()
        n_unmatched = n_total - n_matched

        if n_matched == 0:
            raise ValueError("No matched observations for reprojection error calculation")

        matched_img_df = self.image_points.df[combined_mask]
        matched_obj_indices = self.img_to_obj_map[combined_mask]

        # 2. Prepare arrays for core function
        camera_indices: CameraIndices = np.array(
            [self.camera_array.posed_port_to_index[port] for port in matched_img_df["port"]], dtype=np.int16
        )
        image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
        world_coords: WorldCoords = self.world_points.points[matched_obj_indices]

        # 3. Compute reprojection errors
        errors_xy: ErrorsXY = reprojection_errors(
            self.camera_array, camera_indices, image_coords, world_coords, use_normalized=False
        )

        # 4. Build raw_errors DataFrame
        euclidean_error = np.sqrt(np.sum(errors_xy**2, axis=1))
        raw_errors = pd.DataFrame(
            {
                "sync_index": matched_img_df["sync_index"].values,
                "port": matched_img_df["port"].values,
                "point_id": matched_img_df["point_id"].values,
                "error_x": errors_xy[:, 0],
                "error_y": errors_xy[:, 1],
                "euclidean_error": euclidean_error,
            }
        )

        # 5. Aggregate metrics
        overall_rmse = float(np.sqrt(np.mean(euclidean_error**2)))

        by_camera = {}
        for port in self.camera_array.posed_cameras.keys():
            port_errors = euclidean_error[matched_img_df["port"] == port]
            by_camera[port] = float(np.sqrt(np.mean(port_errors**2))) if len(port_errors) > 0 else 0.0

        by_point_id = {}
        for point_id in np.unique(matched_img_df["point_id"]):
            point_errors = euclidean_error[matched_img_df["point_id"] == point_id]
            by_point_id[point_id] = float(np.sqrt(np.mean(point_errors**2)))

        # 6. Count unmatched by camera (only count for posed cameras)
        unmatched_by_camera = {}
        for port in self.camera_array.cameras.keys():
            port_total = (self.image_points.df["port"] == port).sum()
            port_matched = ((self.image_points.df["port"] == port) & combined_mask).sum()
            unmatched_by_camera[port] = int(port_total - port_matched)

        # 7. Create and cache report
        report = ReprojectionReport(
            overall_rmse=overall_rmse,
            by_camera=by_camera,
            by_point_id=by_point_id,
            n_unmatched_observations=int(n_unmatched),
            unmatched_rate=n_unmatched / n_total if n_total > 0 else 0.0,
            unmatched_by_camera=unmatched_by_camera,
            raw_errors=raw_errors,
            n_observations_matched=int(n_matched),
            n_observations_total=int(n_total),
            n_cameras=len(self.camera_array.posed_cameras),
            n_points=len(self.world_points.points),
        )

        return report

    def optimize(
        self,
        ftol: float = 1e-8,
        max_nfev: int = 1000,
        verbose: int = 2,
    ) -> PointDataBundle:
        """
        Perform bundle adjustment optimization on this PointDataBundle.

        Returns a NEW PointDataBundle with optimized camera parameters and 3D points.
        The original bundle remains unchanged (immutable pattern).
        """
        # Extract static data once - filter to matched observations from posed cameras
        matched_mask = self.img_to_obj_map >= 0
        posed_ports = set(self.camera_array.posed_port_to_index.keys())
        posed_mask: np.ndarray = self.image_points.df["port"].isin(posed_ports).to_numpy()
        combined_mask = matched_mask & posed_mask

        matched_img_df = self.image_points.df[combined_mask]

        camera_indices: CameraIndices = np.array(
            [self.camera_array.posed_port_to_index[port] for port in matched_img_df["port"]], dtype=np.int16
        )

        image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
        image_to_world_indices = self.img_to_obj_map[combined_mask]

        # Initial parameters from current state
        initial_params = self._get_vectorized_params()

        # Get sparsity pattern for Jacobian
        sparsity_pattern = self._get_sparsity_pattern(camera_indices, image_to_world_indices)

        # Perform optimization
        logger.info(f"Beginning bundle adjustment on {len(image_coords)} observations")
        result = least_squares(
            bundle_residuals,
            initial_params,
            args=(self.camera_array, camera_indices, image_coords, image_to_world_indices, True),
            jac_sparsity=sparsity_pattern,  # Now using sparse Jacobian
            verbose=verbose,
            x_scale="jac",
            loss="linear",
            ftol=ftol,
            max_nfev=max_nfev,
            method="trf",
        )

        # Capture optimization status
        termination_reason = _SCIPY_STATUS_REASONS.get(result.status, f"unknown_{result.status}")
        converged = result.status in (1, 2, 3, 4)  # Any gtol/ftol/xtol convergence

        optimization_status = OptimizationStatus(
            converged=converged,
            termination_reason=termination_reason,
            iterations=result.nfev,
            final_cost=float(result.cost),
        )

        # Create new bundle with optimized parameters
        new_camera_array = deepcopy(self.camera_array)
        new_camera_array.update_extrinsic_params(result.x)

        # Extract optimized 3D points
        n_cams = len(self.camera_array.posed_cameras)
        n_cam_params = 6
        optimized_points = result.x[n_cams * n_cam_params :].reshape((-1, 3))

        # Create new world points with optimized coordinates
        new_world_df = self.world_points.df.copy()
        matched_obj_unique = np.unique(image_to_world_indices)
        new_world_df.loc[matched_obj_unique, ["x_coord", "y_coord", "z_coord"]] = optimized_points

        new_world_points = WorldPoints(new_world_df)

        return PointDataBundle(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
            _optimization_status=optimization_status,
        )

    def _get_sparsity_pattern(
        self,
        camera_indices: NDArray[np.int16],
        obj_indices: NDArray[np.int32],
    ) -> lil_matrix:
        """
        Generate sparsity pattern for Jacobian matrix.

        Each observation contributes 2 residuals (x_error, y_error).
        Each residual depends on:
        - 6 camera parameters (rotation + translation)
        - 3 point parameters (x, y, z)

        Args:
            camera_indices: (n_observations,) array mapping observations to cameras
            obj_indices: (n_observations,) array mapping observations to 3D points

        Returns:
            sparsity: lil_matrix of shape (n_residuals, n_params)
        """
        n_observations = len(camera_indices)
        n_cameras = len(self.camera_array.posed_cameras)
        n_points = len(self.world_points.points)

        # Jacobian dimensions: 2 residuals per observation
        n_residuals = n_observations * 2
        n_params = n_cameras * 6 + n_points * 3

        sparsity = lil_matrix((n_residuals, n_params), dtype=int)

        # Observation indices (0 to n_observations-1)
        obs_idx = np.arange(n_observations)

        # Camera parameter dependencies (first 6 params per camera)
        for cam_param in range(6):
            param_col = camera_indices * 6 + cam_param
            sparsity[2 * obs_idx, param_col] = 1  # x residual depends on camera param
            sparsity[2 * obs_idx + 1, param_col] = 1  # y residual depends on camera param

        # Point parameter dependencies (3 params per point, after camera params)
        for point_param in range(3):
            param_col = n_cameras * 6 + obj_indices * 3 + point_param
            sparsity[2 * obs_idx, param_col] = 1  # x residual depends on point param
            sparsity[2 * obs_idx + 1, param_col] = 1  # y residual depends on point param

        return sparsity

    def _get_vectorized_params(self) -> NDArray[np.float64]:
        """
        Convert camera extrinsics and 3D points to a flattened optimization vector.
        Shape: (n_cameras*6 + n_points*3,)
        """
        camera_params = self.camera_array.get_extrinsic_params()  # (n_cams, 6)
        if camera_params is None:
            raise ValueError("Camera extrinsic parameters not initialized")
        points_3d = self.world_points.points  # (n_points, 3)

        return np.concatenate([camera_params.ravel(), points_3d.ravel()])

    def _filter_by_reprojection_thresholds(self, thresholds: dict[int, float], min_per_camera: int) -> PointDataBundle:
        """
        Internal: Filter observations using per-camera error thresholds with safety enforcement.

        Args:
            thresholds: dict mapping camera port -> max_error_pixels for that camera
            min_per_camera: minimum observations to preserve per camera

        Returns:
            New PointDataBundle with filtered observations
        """
        # Get reprojection data (cached)
        report = self.reprojection_report
        raw_errors = report.raw_errors

        # Build initial keep mask: error <= threshold for that camera's port
        threshold_series = raw_errors["port"].map(thresholds)
        keep_mask = (raw_errors["euclidean_error"] <= threshold_series).copy()

        # Apply safety: ensure each camera keeps at least min_per_camera observations
        for port in raw_errors["port"].unique():
            camera_idx = raw_errors["port"] == port
            n_keep = keep_mask[camera_idx].sum()
            n_total = camera_idx.sum()

            # If below minimum and we can add more
            if n_keep < min_per_camera and n_keep < n_total:
                # How many we need to add (capped at total available)
                n_needed = min(min_per_camera, n_total) - n_keep

                # Get errors for observations that would be filtered out
                # pandas stubs don't narrow .loc with boolean mask to Series
                filtered_errors = raw_errors.loc[camera_idx & ~keep_mask, "euclidean_error"]

                if len(filtered_errors) >= n_needed:  # type: ignore[arg-type]
                    # Find the error threshold that would keep exactly n_needed more observations
                    threshold_to_add: float = filtered_errors.nsmallest(n_needed).iloc[-1]  # type: ignore[union-attr]

                    # Update mask to keep observations with error <= threshold_to_add
                    keep_mask[camera_idx] = raw_errors.loc[camera_idx, "euclidean_error"] <= threshold_to_add  # type: ignore[index, operator]

        # Get keys of observations to keep
        keep_keys = raw_errors[keep_mask][["sync_index", "port", "point_id"]]

        # Filter image points by merging with keep keys
        filtered_img_df = self.image_points.df.merge(keep_keys, on=["sync_index", "port", "point_id"], how="inner")
        filtered_image_points = ImagePoints(filtered_img_df)

        # Prune orphaned world points (3D points with no observations)
        remaining_3d_keys = filtered_img_df[["sync_index", "point_id"]].drop_duplicates()
        filtered_world_df = self.world_points.df.merge(remaining_3d_keys, on=["sync_index", "point_id"], how="inner")

        filtered_world_points = WorldPoints(filtered_world_df)

        return PointDataBundle(
            camera_array=self.camera_array,
            image_points=filtered_image_points,
            world_points=filtered_world_points,
        )

    def filter_by_absolute_error(self, max_pixels: float, min_per_camera: int = 10) -> PointDataBundle:
        """
        Remove observations with reprojection error > max_pixels.

        Safety: Ensures each camera keeps at least min_per_camera observations.
        If a camera would drop below this threshold, the lowest-error observations
        are restored until the threshold is met.

        Args:
            max_pixels: Maximum reprojection error (pixels) to keep
            min_per_camera: Minimum observations per camera (safety floor)

        Returns:
            New PointDataBundle with filtered observations
        """
        if max_pixels <= 0:
            raise ValueError(f"max_pixels must be positive, got {max_pixels}")

        if min_per_camera < 1:
            raise ValueError(f"min_per_camera must be >= 1, got {min_per_camera}")

        # Build uniform thresholds for all posed cameras
        thresholds = {port: max_pixels for port in self.camera_array.posed_cameras.keys()}

        return self._filter_by_reprojection_thresholds(thresholds, min_per_camera)

    def filter_by_percentile_error(
        self, percentile: float, scope: Literal["per_camera", "overall"] = "per_camera", min_per_camera: int = 10
    ) -> PointDataBundle:
        """
        Remove worst N% of observations based on reprojection error.

        Args:
            percentile: Percentage of worst observations to remove (0-100)
            scope: "per_camera" computes percentile per camera, "overall" uses global percentile
            min_per_camera: Minimum observations per camera (safety floor)

        Returns:
            New PointDataBundle with filtered observations
        """
        if not (0 < percentile <= 100):
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

        if min_per_camera < 1:
            raise ValueError(f"min_per_camera must be >= 1, got {min_per_camera}")

        report = self.reprojection_report
        raw_errors = report.raw_errors

        if scope == "per_camera":
            # Compute (100 - percentile)th percentile per camera
            thresholds: dict[int, float] = {}
            for port in self.camera_array.posed_cameras.keys():
                camera_errors = raw_errors[raw_errors["port"] == port]["euclidean_error"]
                if len(camera_errors) > 0:
                    # Keep the best (100 - percentile) percent
                    keep_percentile = 100 - percentile
                    thresholds[port] = float(np.percentile(camera_errors, keep_percentile))
                else:
                    thresholds[port] = float(np.inf)  # No observations, keep nothing

        elif scope == "overall":
            # Compute global (100 - percentile)th percentile
            keep_percentile = 100 - percentile
            global_threshold = float(np.percentile(raw_errors["euclidean_error"], keep_percentile))
            thresholds = {port: global_threshold for port in self.camera_array.posed_cameras.keys()}

        else:
            raise ValueError(f"scope must be 'per_camera' or 'overall', got {scope}")

        return self._filter_by_reprojection_thresholds(thresholds, min_per_camera)

    def compute_scale_accuracy(self, sync_index: int) -> ScaleAccuracyData:
        """Compute scale accuracy comparing triangulated points to known object geometry.

        Compares triangulated world points at the given sync_index to their
        corresponding ground truth object positions (from obj_loc columns) to
        assess reconstruction accuracy. Uses ALL pairwise distances between
        detected corners for robust statistical measurement.

        Args:
            sync_index: Frame index to compute accuracy at

        Returns:
            ScaleAccuracyData with distance RMSE and relative error

        Raises:
            ValueError: If insufficient matched points at sync_index (< 2)
        """
        # Extract data at sync_index
        img_df = self.image_points.df
        world_df = self.world_points.df

        img_subset = img_df[img_df["sync_index"] == sync_index]
        world_subset = world_df[world_df["sync_index"] == sync_index]

        if img_subset.empty:
            raise ValueError(f"No image observations at sync_index {sync_index}")
        if world_subset.empty:
            raise ValueError(f"No world points at sync_index {sync_index}")

        # Get image points with object locations at reference frame
        # Use drop_duplicates on img_subset since multiple cameras may see same point_id
        obj_points_df = img_subset[["point_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]].drop_duplicates(
            subset=["point_id"]
        )

        # Merge world points with object locations by point_id
        merged = world_subset.merge(obj_points_df, on="point_id", how="inner")

        if len(merged) < 2:
            raise ValueError(f"Insufficient matched points for scale accuracy: {len(merged)} (need at least 2)")

        # Handle planar objects (z=0 or NaN)
        if merged["obj_loc_z"].isna().all():
            merged = merged.copy()
            merged["obj_loc_z"] = 0.0

        # Filter out any remaining NaN values
        valid_mask = ~merged[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].isna().any(axis=1)
        merged = merged[valid_mask]

        if len(merged) < 2:
            raise ValueError("Insufficient valid points after NaN filtering (need at least 2)")

        # Extract arrays for scale accuracy computation
        world_points = merged[["x_coord", "y_coord", "z_coord"]].to_numpy()
        object_points = merged[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].to_numpy()

        return compute_scale_accuracy_impl(world_points, object_points, sync_index)

    def align_to_object(self, sync_index: int) -> "PointDataBundle":
        """
        Align the bundle to real-world units using object point correspondences.

        Uses the 3D points triangulated at the given sync_index and their
        corresponding ground truth object positions (from obj_loc columns) to
        estimate a similarity transform that scales the reconstruction to real-world units.

        Note:
            Object coordinates (obj_loc_*) must be in real-world units (typically meters).
            For Charuco boards, this requires defining the board with square_length in meters.

        For planar Charuco boards, obj_loc_z may be missing and will be treated as 0.
        The obj_loc coordinates must be in the target units (typically meters).

        Args:
            sync_index: Frame index where object is visible and has obj_loc data

        Returns:
            New PointDataBundle with cameras and world points in object coordinate units

        Raises:
            ValueError: If insufficient valid correspondences (< 3 points) or missing data
        """
        # Extract data at sync_index
        img_df = self.image_points.df
        world_df = self.world_points.df

        img_subset = img_df[img_df["sync_index"] == sync_index]
        world_subset = world_df[world_df["sync_index"] == sync_index]

        if img_subset.empty:
            raise ValueError(f"No image observations at sync_index {sync_index}")
        if world_subset.empty:
            raise ValueError(f"No world points at sync_index {sync_index}")

        # Merge on point_id to find correspondences
        merged = pd.merge(
            world_subset[["point_id", "x_coord", "y_coord", "z_coord"]],
            img_subset[["point_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]],
            on="point_id",
            how="inner",
        )

        if len(merged) < 3:
            raise ValueError(f"Need at least 3 point correspondences at sync_index {sync_index}, got {len(merged)}")

        # Handle missing obj_loc_z (planar boards)
        if merged["obj_loc_z"].isna().all():
            logger.info("obj_loc_z is all NaN, assuming planar board with z=0")
            merged["obj_loc_z"] = 0.0

        # Filter out any rows with NaN object coordinates
        obj_cols = ["obj_loc_x", "obj_loc_y", "obj_loc_z"]
        valid_mask = ~merged[obj_cols].isna().any(axis=1)
        merged = merged[valid_mask]

        if len(merged) < 3:
            raise ValueError(
                f"Need at least 3 valid point correspondences at sync_index {sync_index}, "
                f"got {len(merged)} after filtering NaN values"
            )

        # Prepare source (triangulated) and target (object) points
        source_points = merged[["x_coord", "y_coord", "z_coord"]].values.astype(np.float64)
        target_points = merged[obj_cols].values.astype(np.float64)

        # Estimate and apply transform
        transform = estimate_similarity_transform(source_points, target_points)

        logger.info(
            f"Estimated alignment: scale={transform.scale:.6f}, "
            f"translation={transform.translation}, rotation_det={np.linalg.det(transform.rotation):.6f}"
        )

        new_camera_array, new_world_points = apply_similarity_transform(self.camera_array, self.world_points, transform)

        return PointDataBundle(
            camera_array=new_camera_array, image_points=self.image_points, world_points=new_world_points
        )

    @property
    def unique_sync_indices(self) -> np.ndarray:
        """
        Return sorted array of unique sync_index values present in world_points.

        Used for slider range in visualization widgets.
        """
        indices = self.world_points.df["sync_index"].unique()
        return np.sort(indices)

    def rotate(self, axis: Literal["x", "y", "z"], angle_degrees: float) -> "PointDataBundle":
        """
        Rotate the coordinate system around the specified axis.

        Uses right-hand rule: positive angle = counter-clockwise rotation
        when looking down the positive axis toward the origin.

        Transforms both camera extrinsics and world points, returning a new
        immutable bundle. The original bundle remains unchanged.

        Args:
            axis: The axis to rotate around ("x", "y", or "z")
            angle_degrees: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            New PointDataBundle with rotated coordinate system.
        """
        angle_rad = np.radians(angle_degrees)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # Standard rotation matrices following right-hand rule
        if axis == "x":
            rotation = np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ],
                dtype=np.float64,
            )
        elif axis == "y":
            rotation = np.array(
                [
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c],
                ],
                dtype=np.float64,
            )
        elif axis == "z":
            rotation = np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")

        transform = SimilarityTransform(
            rotation=rotation,
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
        )

        new_camera_array, new_world_points = apply_similarity_transform(self.camera_array, self.world_points, transform)

        return PointDataBundle(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
        )


if __name__ == "__main__":
    from pathlib import Path
    from caliscope import __root__
    from caliscope.core.point_data import ImagePoints
    from caliscope.core.point_data_bundle import PointDataBundle
    from caliscope import persistence

    # Load test data
    session_path = Path(__root__, "tests", "sessions", "larger_calibration_post_monocal")
    xy_path = session_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
    array_path = session_path / "camera_array.toml"

    image_points = ImagePoints.from_csv(xy_path)
    camera_array = persistence.load_camera_array(array_path)
    world_points = image_points.triangulate(camera_array)

    bundle = PointDataBundle(camera_array, image_points, world_points)

    # Inspect the reprojection report
    report = bundle.reprojection_report
