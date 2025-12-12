from __future__ import annotations

import cv2
import numpy as np
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
    obj_indices: np.ndarray = None  # Computed in __post_init__, -1 for unmatched

    def __post_init__(self):
        """Compute mapping and validate geometry."""
        if self.obj_indices is None:
            # Compute mapping without side effects
            obj_indices = self._compute_obj_indices()
            object.__setattr__(self, "obj_indices", obj_indices)

        self._validate_geometry()

    def _compute_obj_indices(self) -> np.ndarray:
        """Map each image observation to its world point index. Returns -1 for unmatched."""
        # Create mapping from (sync_index, point_id) -> world point index
        world_df = self.world_points.df.reset_index().rename(columns={"index": "world_idx"})
        mapping = world_df.set_index(["sync_index", "point_id"])["world_idx"].to_dict()

        # Map each image observation
        img_df = self.image_points.df
        keys = list(zip(img_df["sync_index"], img_df["point_id"]))
        obj_indices = np.array([mapping.get(key, -1) for key in keys], dtype=np.int32)

        n_unmatched = np.sum(obj_indices == -1)
        if n_unmatched > 0:
            logger.info(f"{n_unmatched} of {len(obj_indices)} image observations have no world point")

        return obj_indices

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
        n_matched = np.sum(self.obj_indices >= 0)
        if n_matched == 0:
            raise ValueError("No image observations have corresponding world points")

        if n_matched < n_world * 2:
            logger.warning(
                f"Suspicious geometry: {n_matched} matched observations for {n_world} world points. "
                f"Expected at least {n_world * 2} for multi-view geometry."
            )

        # Validate indices are in bounds
        valid_indices = self.obj_indices[self.obj_indices >= 0]
        if valid_indices.size > 0 and valid_indices.max() >= n_world:
            raise ValueError(f"obj_indices contains out-of-bounds index: {valid_indices.max()} >= {n_world}")

    def calculate_reprojection_error(self, normalized: bool = False) -> float:
        """Calculate RMSE using only matched observations."""
        # Filter to matched observations at use time
        matched_mask = self.obj_indices >= 0
        if not matched_mask.any():
            logger.error("No matched observations for RMSE calculation")
            return float("inf")

        matched_img_df = self.image_points.df[matched_mask]
        matched_obj_indices = self.obj_indices[matched_mask]
        world_coords = self.world_points.points[matched_obj_indices]

        total_squared_error = 0.0
        total_observations = 0

        for port, camera_data in self.camera_array.posed_cameras.items():
            cam_mask = matched_img_df["port"] == port
            if not cam_mask.any():
                continue

            n_obs = cam_mask.sum()
            cam_world_coords = world_coords[cam_mask.values]
            cam_observed = matched_img_df.loc[cam_mask, ["img_loc_x", "img_loc_y"]].values

            # ... rest of projection logic unchanged ...
            if normalized:
                cam_observed = camera_data.undistort_points(cam_observed)
                cam_matrix = np.identity(3)
                dist_coeffs = None
            else:
                cam_matrix = camera_data.matrix
                dist_coeffs = camera_data.distortions

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

        return np.sqrt(total_squared_error / total_observations)

    @property
    def point_estimates(self) -> PointEstimates:
        """Convert to PointEstimates format for bundle adjustment."""
        # Filter to matched observations only
        matched_mask = self.obj_indices >= 0
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
        matched_obj_indices = self.obj_indices[matched_mask]

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
