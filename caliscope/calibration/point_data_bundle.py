from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.point_data import ImagePoints, WorldPoints
from caliscope.calibration.capture_volume.point_estimates import PointEstimates

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class BundleMetadata:
    """
    Complete provenance tracking for a PointDataBundle.

    This metadata captures how the bundle was created and all operations
    performed on it, enabling reproducibility and debugging.
    """

    created_at: str
    generation_method: Literal["triangulation", "bundle_adjustment"]
    generation_params: dict
    camera_array_path: Path
    operations: list[dict] = field(default_factory=list)
    source_files: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class PointDataBundle:
    camera_array: CameraArray
    image_points: ImagePoints
    world_points: WorldPoints
    metadata: BundleMetadata
    obj_indices: np.ndarray = None  # Internal mapping, computed in __post_init__

    def __post_init__(self):
        """Validate data and compute mapping."""
        if self.obj_indices is None:
            object.__setattr__(self, "obj_indices", self._compute_obj_indices())

        # Validate counts make geometric sense
        self._validate_geometry()

    def _compute_obj_indices(self) -> np.ndarray:
        """Map each image observation to its world point index."""
        # Create mapping from identifiers to world point indices
        world_df = self.world_points.df.reset_index().rename(columns={"index": "obj_idx"})
        mapping = world_df.set_index(["sync_index", "point_id"])["obj_idx"].to_dict()

        # Map image observations
        img_df = self.image_points.df
        img_keys = list(zip(img_df["sync_index"], img_df["point_id"]))

        obj_indices = np.array([mapping.get(key, -1) for key in img_keys], dtype=np.int32)

        # Filter out unmatched observations
        valid_mask = obj_indices >= 0

        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(
                f"Filtering {n_invalid} image observations without world points (out of {len(img_df)} total)"
            )

            # Update attributes with filtered data
            filtered_img_df = img_df[valid_mask].copy()
            filtered_image_points = ImagePoints(filtered_img_df)
            filtered_obj_indices = obj_indices[valid_mask]

            object.__setattr__(self, "image_points", filtered_image_points)
            object.__setattr__(self, "obj_indices", filtered_obj_indices)

        return self.obj_indices

    def _validate_geometry(self):
        """Ensure data counts make geometric sense."""
        n_img = len(self.image_points.df)
        n_world = len(self.world_points.df)
        n_cams = len(self.camera_array.posed_cameras)

        # Basic sanity checks
        if n_img == 0:
            raise ValueError("No image observations provided")
        if n_world == 0:
            raise ValueError("No world points provided")
        if n_cams == 0:
            raise ValueError("No posed cameras in array")

        # Geometry check: should have more observations than world points
        if n_img < n_world * 2:
            logger.warning(
                f"Suspicious geometry: {n_img} observations for {n_world} world points. "
                f"Expected at least {n_world * 2} for multi-view geometry."
            )

        # Validate mapping
        if len(self.obj_indices) != n_img:
            raise ValueError("obj_indices length mismatch")

        if self.obj_indices.max() >= n_world:
            raise ValueError("obj_indices contains out-of-bounds world point index")

    def calculate_reprojection_error(self, normalized: bool = False) -> float:
        """Calculate RMSE using the internal mapping."""
        total_squared_error = 0.0
        total_observations = 0

        # Pre-fetch world coordinates for all observations using mapping
        world_coords = self.world_points.points[self.obj_indices]

        for port, camera_data in self.camera_array.posed_cameras.items():
            # Get camera-specific observations
            cam_mask = self.image_points.df["port"] == port
            if not cam_mask.any():
                continue

            n_obs = cam_mask.sum()
            cam_world_coords = world_coords[cam_mask.values]
            cam_observed = self.image_points.df.loc[cam_mask, ["img_loc_x", "img_loc_y"]].values

            # Handle normalized vs pixel coordinates
            if normalized:
                cam_observed = camera_data.undistort_points(cam_observed)
                cam_matrix = np.identity(3)
                dist_coeffs = None
            else:
                cam_matrix = camera_data.matrix
                dist_coeffs = camera_data.distortions

            # Project and calculate error
            projected, _ = cv2.projectPoints(
                cam_world_coords.reshape(-1, 1, 3),
                camera_data.rotation,
                camera_data.translation,
                cam_matrix,
                dist_coeffs,
            )
            projected = projected.reshape(-1, 2)

            errors = np.sum((projected - cam_observed) ** 2, axis=1)
            total_squared_error += np.sum(errors)
            total_observations += n_obs

        if total_observations == 0:
            logger.error("No valid observations for RMSE calculation!")
            return float("inf")

        return np.sqrt(total_squared_error / total_observations)

    def filter_worst_fraction(
        self,
        fraction: float,
        strategy: Literal["global", "per_camera"],
        min_observations_per_camera: int = 20,
        level: Literal["observation", "point"] = "observation",
    ) -> PointDataBundle:
        """
        Remove worst-fitting observations and re-triangulate.

        This is the primary method for iterative refinement of calibration data.
        It removes outliers based on reprojection error, then re-triangulates
        the remaining points to create a new, cleaner bundle.

        Args:
            fraction: Remove worst-fitting fraction (e.g., 0.05 = worst 5%)
            strategy: "global" removes worst overall; "per_camera" removes worst per camera
            min_observations_per_camera: Abort if any camera would drop below this threshold
            level: "observation" removes individual 2D detections; "point" removes entire 3D points

        Returns:
            New PointDataBundle with filtered data and updated provenance

        Raises:
            ValueError: If filtering would leave any camera with insufficient observations
        """
        if not 0 < fraction < 1:
            raise ValueError(f"Fraction must be between 0 and 1, got {fraction}")

        # Calculate per-observation errors
        errors = self._calculate_per_observation_errors()

        # Create mask for observations to keep
        keep_mask = self._create_keep_mask(errors, fraction, strategy, level)

        # Apply guardrails
        self._validate_guardrails(keep_mask, min_observations_per_camera)

        # Filter image points
        filtered_image_df = self.image_points.df[keep_mask].copy()
        filtered_image_points = ImagePoints(filtered_image_df)

        filtered_world_points = filtered_image_points.triangulate(self.camera_array)

        # Update metadata with operation history
        new_operations = self.metadata.operations + [
            {
                "type": "filter",
                "params": {
                    "fraction": fraction,
                    "strategy": strategy,
                    "min_observations_per_camera": min_observations_per_camera,
                    "level": level,
                },
                "timestamp": pd.Timestamp.now().isoformat(),
                "original_observations": len(self.image_points.df),
                "remaining_observations": len(filtered_image_df),
            }
        ]

        new_metadata = BundleMetadata(
            created_at=self.metadata.created_at,
            generation_method=self.metadata.generation_method,
            generation_params=self.metadata.generation_params,
            camera_array_path=self.metadata.camera_array_path,
            operations=new_operations,
            source_files=self.metadata.source_files,
        )

        return PointDataBundle(
            camera_array=self.camera_array,
            image_points=filtered_image_points,
            world_points=filtered_world_points,
            metadata=new_metadata,
        )

    def _calculate_per_observation_errors(self) -> pd.Series:
        """Calculate reprojection error for each observation"""
        errors = pd.Series(index=self.image_points.df.index, dtype=float)

        for port, camera_data in self.camera_array.cameras.items():
            if camera_data.ignore or camera_data.rotation is None:
                continue

            camera_observations = self.image_points.df[self.image_points.df["port"] == port]

            if camera_observations.empty:
                continue

            # Get world points for these observations
            point_ids = camera_observations["point_id"].values
            sync_indices = camera_observations["sync_index"].values

            # Match world points by point_id and sync_index
            world_df = self.world_points.df
            world_mask = world_df["point_id"].isin(point_ids) & world_df["sync_index"].isin(sync_indices)
            world_coords = world_df[world_mask][["x_coord", "y_coord", "z_coord"]].values

            if len(world_coords) != len(camera_observations):
                # Handle mismatched data gracefully
                continue

            # Project and calculate errors
            projected_points, _ = cv2.projectPoints(
                world_coords.reshape(-1, 1, 3),
                camera_data.rotation,
                camera_data.translation,
                camera_data.matrix,
                camera_data.distortions,
            )
            projected_points = projected_points.reshape(-1, 2)

            observed_points = camera_observations[["img_loc_x", "img_loc_y"]].values

            camera_errors = np.sqrt(np.sum((projected_points - observed_points) ** 2, axis=1))
            errors.loc[camera_observations.index] = camera_errors

        return errors

    def _create_keep_mask(self, errors: pd.Series, fraction: float, strategy: str, level: str) -> pd.Series:
        """Create boolean mask for observations to keep"""
        keep_mask = pd.Series(True, index=self.image_points.df.index)

        if strategy == "global":
            # Remove worst fraction globally
            threshold = errors.quantile(1 - fraction)
            keep_mask = errors <= threshold

        elif strategy == "per_camera":
            # Remove worst fraction per camera
            for port in self.image_points.df["port"].unique():
                camera_mask = self.image_points.df["port"] == port
                camera_errors = errors[camera_mask]

                if len(camera_errors) > 0:
                    threshold = camera_errors.quantile(1 - fraction)
                    keep_mask.loc[camera_mask] = camera_errors <= threshold

        if level == "point":
            # If any observation of a point is bad, remove all observations of that point
            bad_point_ids = self.image_points.df.loc[~keep_mask, "point_id"].unique()
            keep_mask = ~self.image_points.df["point_id"].isin(bad_point_ids)

        return keep_mask

    def _validate_guardrails(self, keep_mask: pd.Series, min_observations: int):
        """Ensure no camera drops below minimum observations"""
        remaining_counts = self.image_points.df[keep_mask]["port"].value_counts()

        for port in self.image_points.df["port"].unique():
            if port not in remaining_counts or remaining_counts[port] < min_observations:
                raise ValueError(
                    f"Filtering would leave camera {port} with insufficient "
                    f"observations (min: {min_observations}). Aborting filter."
                )

    def error_breakdown(self) -> dict[str, pd.DataFrame]:
        """
        Return detailed error statistics for diagnostics.

        Returns:
            Dictionary containing three DataFrames:
            - "by_camera": Error statistics per camera
            - "by_point": Error statistics per 3D point
            - "by_observation": Individual observation errors
        """
        # Calculate per-observation errors
        observation_errors = self._calculate_per_observation_errors()

        # Create observation-level DataFrame
        observation_df = self.image_points.df.copy()
        observation_df["reprojection_error"] = observation_errors

        # Group by camera
        by_camera = (
            observation_df.groupby("port")
            .agg(
                {
                    "reprojection_error": ["mean", "std", "count"],
                    "img_loc_x": "count",  # Total observations per camera
                }
            )
            .round(6)
        )
        by_camera.columns = ["mean_error", "std_error", "observation_count"]
        by_camera = by_camera.reset_index()

        # Group by point
        by_point = observation_df.groupby("point_id").agg({"reprojection_error": ["mean", "std", "count"]}).round(6)
        by_point.columns = ["mean_error", "std_error", "observation_count"]
        by_point = by_point.reset_index()

        return {
            "by_camera": by_camera,
            "by_point": by_point,
            "by_observation": observation_df[["sync_index", "port", "point_id", "reprojection_error"]],
        }

    @property
    def point_estimates(self) -> PointEstimates:
        """
        Convert bundle data to PointEstimates format for bundle adjustment.

        Creates proper mapping between 2D observations and 3D points without
        relying on circular references to ImagePoints or CameraArray.
        """
        # Get image points data
        img_data = self.image_points.df

        # Get world points data with explicit index
        world_df = self.world_points.df.reset_index(drop=True)
        world_df["xyz_index"] = world_df.index

        # Merge to align 2D observations with 3D points
        merged = img_data.merge(
            world_df[["sync_index", "point_id", "xyz_index"]], on=["sync_index", "point_id"], how="inner"
        )

        # Filter to posed cameras only
        posed_ports = list(self.camera_array.posed_port_to_index.keys())
        merged = merged[merged["port"].isin(posed_ports)]

        if merged.empty:
            logger.warning("No merged 2D-3D observations after filtering")
            return PointEstimates(
                sync_indices=np.array([], dtype=np.int64),
                camera_indices=np.array([], dtype=np.int64),
                point_id=np.array([], dtype=np.int64),
                img=np.array([], dtype=np.float32).reshape(0, 2),
                obj_indices=np.array([], dtype=np.int64),
                obj=np.array([], dtype=np.float32).reshape(0, 3),
            )

        # Prune orphaned 3D points and create compact mapping
        used_xyz_indices = merged["xyz_index"].unique()
        used_xyz_indices.sort()
        obj = world_df.loc[used_xyz_indices, ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)

        # Create mapping from old (world_df) indices to new (compact) indices
        old_to_new_map = {old: new for new, old in enumerate(used_xyz_indices)}
        merged["obj_index_pruned"] = merged["xyz_index"].map(old_to_new_map)

        # Map ports to camera indices for optimization
        merged["camera_index"] = merged["port"].map(self.camera_array.posed_port_to_index)

        # Extract arrays for PointEstimates
        sync_indices = merged["sync_index"].to_numpy(dtype=np.int64)
        camera_indices = merged["camera_index"].to_numpy(dtype=np.int64)
        point_id = merged["point_id"].to_numpy(dtype=np.int64)
        img = merged[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32)
        obj_indices = merged["obj_index_pruned"].to_numpy(dtype=np.int64)

        # Validate consistency
        assert len(camera_indices) == len(img) == len(obj_indices), "Mismatch in 2D data array lengths"
        if len(obj_indices) > 0:
            assert obj_indices.max() < obj.shape[0], "CRITICAL: obj_indices contains an out-of-bounds index"
            assert np.unique(obj_indices).size == obj.shape[0], "CRITICAL: Orphaned 3D points detected!"

        return PointEstimates(
            sync_indices=sync_indices,
            camera_indices=camera_indices,
            point_id=point_id,
            img=img,
            obj_indices=obj_indices,
            obj=obj,
        )
