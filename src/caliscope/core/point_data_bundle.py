from __future__ import annotations
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from copy import deepcopy
from numpy.typing import NDArray

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.point_data import ImagePoints, WorldPoints
from caliscope.core.capture_volume.point_estimates import PointEstimates
from caliscope.core.reprojection import (
    ErrorsXY,
    reprojection_errors,
    bundle_residuals,
    ImageCoords,
    WorldCoords,
    CameraIndices,
)
from caliscope.core.reprojection_report import ReprojectionReport
import pandas as pd

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class PointDataBundle:
    camera_array: CameraArray
    image_points: ImagePoints
    world_points: WorldPoints
    img_to_obj_map: np.ndarray = None  # Renamed from obj_indices

    def __post_init__(self):
        """Compute mapping and validate geometry."""
        if self.img_to_obj_map is None:
            img_to_obj_map = self._compute_img_to_obj_map()
            object.__setattr__(self, "img_to_obj_map", img_to_obj_map)
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

    @property
    def reprojection_report(self) -> ReprojectionReport:
        """
        Generate comprehensive reprojection error report in pixel units.
        Caches result for subsequent calls since bundle data is immutable.
        """
        # Check cache first
        if hasattr(self, "_reprojection_report"):
            return self._reprojection_report

        # 1. Filter to matched observations
        matched_mask = self.img_to_obj_map >= 0
        n_total = len(self.img_to_obj_map)
        n_matched = matched_mask.sum()
        n_unmatched = n_total - n_matched

        if n_matched == 0:
            raise ValueError("No matched observations for reprojection error calculation")

        matched_img_df = self.image_points.df[matched_mask]
        matched_obj_indices = self.img_to_obj_map[matched_mask]

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

        # 6. Count unmatched by camera
        unmatched_by_camera = {}
        for port in self.camera_array.cameras.keys():
            port_total = (self.image_points.df["port"] == port).sum()
            port_matched = ((self.image_points.df["port"] == port) & matched_mask).sum()
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

        object.__setattr__(self, "_reprojection_report", report)
        return report

    def optimize(
        self,
        ftol: float = 1e-8,
        max_nfev: int | None = None,
        verbose: int = 2,
    ) -> PointDataBundle:
        """
        Perform bundle adjustment optimization on this PointDataBundle.

        Returns a NEW PointDataBundle with optimized camera parameters and 3D points.
        The original bundle remains unchanged (immutable pattern).
        """
        # Extract static data once
        matched_mask = self.img_to_obj_map >= 0
        matched_img_df = self.image_points.df[matched_mask]

        camera_indices: CameraIndices = np.array(
            [self.camera_array.posed_port_to_index[port] for port in matched_img_df["port"]], dtype=np.int16
        )

        image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
        image_to_world_indices = self.img_to_obj_map[matched_mask]

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
        points_3d = self.world_points.points  # (n_points, 3)

        return np.concatenate([camera_params.ravel(), points_3d.ravel()])

    @property
    def point_estimates(self) -> PointEstimates:
        """Convert to PointEstimates format for bundle adjustment.
        Currently a legacy conversion that will be phased out"""
        # Filter to matched observations only
        matched_mask = self.img_to_obj_map >= 0
        if not matched_mask.any():
            logger.warning("No matched observations for PointEstimates conversion")
            return PointEstimates(
                sync_indices=np.array([], dtype=np.int32),
                camera_indices=np.array([], dtype=np.int16),
                point_id=np.array([], dtype=np.int16),
                img=np.array([], dtype=np.float64).reshape(0, 2),
                obj_indices=np.array([], dtype=np.int32),
                obj=np.array([], dtype=np.float64).reshape(0, 3),
            )

        matched_img_df = self.image_points.df[matched_mask]
        matched_obj_indices = self.img_to_obj_map[matched_mask]

        # Create compact world points array (unique points only)
        unique_obj_indices = np.unique(matched_obj_indices)
        world_points_compact = self.world_points.points[unique_obj_indices]

        # Map to compact indices
        index_map = {old: new for new, old in enumerate(unique_obj_indices)}
        compact_obj_indices = np.array([index_map[idx] for idx in matched_obj_indices], dtype=np.int32)

        # Map ports to camera indices
        port_to_index = self.camera_array.posed_port_to_index
        camera_indices = np.array([port_to_index[port] for port in matched_img_df["port"]], dtype=np.int16)

        return PointEstimates(
            sync_indices=matched_img_df["sync_index"].to_numpy(dtype=np.int32),
            camera_indices=camera_indices,
            point_id=matched_img_df["point_id"].to_numpy(dtype=np.int16),
            img=matched_img_df[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float64),
            obj_indices=compact_obj_indices,
            obj=world_points_compact,
        )

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
                filtered_errors = raw_errors.loc[camera_idx & ~keep_mask, "euclidean_error"]

                if len(filtered_errors) >= n_needed:
                    # Find the error threshold that would keep exactly n_needed more observations
                    threshold_to_add = filtered_errors.nsmallest(n_needed).iloc[-1]

                    # Update mask to keep observations with error <= threshold_to_add
                    keep_mask[camera_idx] = raw_errors.loc[camera_idx, "euclidean_error"] <= threshold_to_add

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
            thresholds = {}
            for port in self.camera_array.posed_cameras.keys():
                camera_errors = raw_errors[raw_errors["port"] == port]["euclidean_error"]
                if len(camera_errors) > 0:
                    # Keep the best (100 - percentile) percent
                    keep_percentile = 100 - percentile
                    thresholds[port] = np.percentile(camera_errors, keep_percentile)
                else:
                    thresholds[port] = np.inf  # No observations, keep nothing

        elif scope == "overall":
            # Compute global (100 - percentile)th percentile
            keep_percentile = 100 - percentile
            global_threshold = np.percentile(raw_errors["euclidean_error"], keep_percentile)
            thresholds = {port: global_threshold for port in self.camera_array.posed_cameras.keys()}

        else:
            raise ValueError(f"scope must be 'per_camera' or 'overall', got {scope}")

        return self._filter_by_reprojection_thresholds(thresholds, min_per_camera)


if __name__ == "__main__":
    from pathlib import Path
    from caliscope import __root__
    from caliscope.post_processing.point_data import ImagePoints
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
