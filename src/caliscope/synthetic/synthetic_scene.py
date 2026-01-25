"""Complete synthetic calibration scenario combining cameras, object, and trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import strip_extrinsics
from caliscope.core.coverage_analysis import compute_coverage_matrix
from caliscope.synthetic.filter_config import FilterConfig
from caliscope.synthetic.trajectory import Trajectory


@dataclass(frozen=True)
class SyntheticScene:
    """Complete synthetic calibration scenario with derived data.

    Combines camera rig + calibration object + trajectory to produce
    ground truth world points and image points. All derived data is
    computed lazily and cached.

    Attributes:
        camera_array: Fully posed cameras (ground truth extrinsics)
        calibration_object: Rigid body with known point geometry
        trajectory: SE3 poses of object across frames
        pixel_noise_sigma: Standard deviation of Gaussian noise added to projections
        random_seed: Seed for noise generation (reproducibility)
    """

    camera_array: CameraArray
    calibration_object: CalibrationObject
    trajectory: Trajectory
    pixel_noise_sigma: float = 0.5
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate scene configuration."""
        if self.pixel_noise_sigma < 0:
            raise ValueError(f"pixel_noise_sigma must be >= 0, got {self.pixel_noise_sigma}")

        # Ensure all cameras have extrinsics
        unposed = self.camera_array.unposed_cameras
        if unposed:
            raise ValueError(f"All cameras must have extrinsics. Unposed: {list(unposed.keys())}")

    @cached_property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return len(self.trajectory)

    @cached_property
    def n_cameras(self) -> int:
        """Number of cameras."""
        return len(self.camera_array.cameras)

    @cached_property
    def world_points(self) -> WorldPoints:
        """All object points transformed to world frame across all frames.

        Returns WorldPoints with one row per (frame, point) combination.
        """
        rows = []

        for frame in range(self.n_frames):
            world_coords = self.trajectory.world_points_at_frame(self.calibration_object, frame)

            for i, point_id in enumerate(self.calibration_object.point_ids):
                rows.append(
                    {
                        "sync_index": frame,
                        "point_id": int(point_id),
                        "x_coord": world_coords[i, 0],
                        "y_coord": world_coords[i, 1],
                        "z_coord": world_coords[i, 2],
                        "frame_time": frame / 30.0,  # Assume 30 fps
                    }
                )

        df = pd.DataFrame(rows)
        return WorldPoints(df)

    @cached_property
    def image_points_perfect(self) -> ImagePoints:
        """Perfect projections (no noise). Includes obj_loc_x/y/z columns.

        Projects all world points through all cameras, filtering to those
        that fall within image bounds.
        """
        return self._project_to_cameras(add_noise=False)

    @cached_property
    def image_points_noisy(self) -> ImagePoints:
        """Projections with Gaussian noise added."""
        return self._project_to_cameras(add_noise=True)

    def _project_to_cameras(self, add_noise: bool) -> ImagePoints:
        """Project world points to all cameras."""
        rng = np.random.default_rng(self.random_seed) if add_noise else None

        rows = []

        for frame in range(self.n_frames):
            # Get object points in world coordinates
            world_coords = self.trajectory.world_points_at_frame(self.calibration_object, frame)

            # Object-local coordinates (for obj_loc columns)
            obj_local = self.calibration_object.points

            for port, camera in self.camera_array.cameras.items():
                if camera.rotation is None or camera.translation is None:
                    continue
                if camera.matrix is None or camera.distortions is None:
                    continue

                # Project points using OpenCV
                projected, _ = cv2.projectPoints(
                    world_coords.reshape(-1, 1, 3),
                    camera.rotation,
                    camera.translation,
                    camera.matrix,
                    camera.distortions,
                )
                projected = projected.reshape(-1, 2)

                # Add noise if requested
                if add_noise and rng is not None:
                    noise = rng.normal(0, self.pixel_noise_sigma, projected.shape)
                    projected = projected + noise

                # Filter to points within image bounds
                w, h = camera.size
                for i, point_id in enumerate(self.calibration_object.point_ids):
                    x, y = projected[i]

                    if 0 <= x < w and 0 <= y < h:
                        rows.append(
                            {
                                "sync_index": frame,
                                "port": port,
                                "point_id": int(point_id),
                                "img_loc_x": float(x),
                                "img_loc_y": float(y),
                                "obj_loc_x": obj_local[i, 0],
                                "obj_loc_y": obj_local[i, 1],
                                "obj_loc_z": obj_local[i, 2],
                                "frame_time": frame / 30.0,
                            }
                        )

        df = pd.DataFrame(rows)
        return ImagePoints(df)

    @cached_property
    def coverage_matrix(self) -> NDArray[np.int64]:
        """(n_cameras, n_cameras) matrix of shared observation counts.

        Element [i, j] is the number of (frame, point) pairs visible
        from both camera i and camera j. Diagonal is total observations per camera.
        """
        port_to_index = {port: idx for idx, port in enumerate(sorted(self.camera_array.cameras.keys()))}
        return compute_coverage_matrix(self.image_points_noisy, port_to_index)

    def intrinsics_only_cameras(self) -> CameraArray:
        """Return cameras with extrinsics stripped (for pipeline input)."""
        return strip_extrinsics(self.camera_array)

    def apply_filter(self, config: FilterConfig) -> ImagePoints:
        """Apply filters to noisy image points. Returns filtered copy."""
        return config.apply(self.image_points_noisy)
