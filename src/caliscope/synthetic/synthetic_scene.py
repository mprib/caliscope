"""Complete synthetic calibration scenario combining cameras, objects, and trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.coverage_analysis import compute_coverage_matrix
from caliscope.core.point_data import STATIC_SYNC_INDEX, ImagePoints, WorldPoints
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import strip_extrinsics
from caliscope.synthetic.filter_config import FilterConfig
from caliscope.synthetic.trajectory import Trajectory


@dataclass(frozen=True)
class SceneObject:
    """A single rigid object moving through the scene, tied to a trajectory.

    static=True collapses the object to a single set of world points at
    STATIC_SYNC_INDEX, matching how CaptureVolume.bootstrap() and
    ImagePoints.triangulate() treat static markers.
    """

    object_id: int
    calibration_object: CalibrationObject
    trajectory: Trajectory
    static: bool = False


@dataclass(frozen=True)
class SyntheticScene:
    """Complete synthetic calibration scenario with derived data.

    Supports one or many independently-moving SceneObjects sharing a
    camera rig. All derived data is computed lazily and cached.
    """

    camera_array: CameraArray
    objects: tuple[SceneObject, ...]
    pixel_noise_sigma: float = 0.5
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.pixel_noise_sigma < 0:
            raise ValueError(f"pixel_noise_sigma must be >= 0, got {self.pixel_noise_sigma}")

        unposed = self.camera_array.unposed_cameras
        if unposed:
            raise ValueError(f"All cameras must have extrinsics. Unposed: {list(unposed.keys())}")

        if not self.objects:
            raise ValueError("At least one SceneObject required")

    @classmethod
    def single(
        cls,
        camera_array: CameraArray,
        calibration_object: CalibrationObject,
        trajectory: Trajectory,
        pixel_noise_sigma: float = 0.5,
        random_seed: int = 42,
    ) -> SyntheticScene:
        """Convenience constructor for single-object scenes."""
        return cls(
            camera_array=camera_array,
            objects=(
                SceneObject(
                    object_id=0,
                    calibration_object=calibration_object,
                    trajectory=trajectory,
                ),
            ),
            pixel_noise_sigma=pixel_noise_sigma,
            random_seed=random_seed,
        )

    @cached_property
    def n_frames(self) -> int:
        return max(len(obj.trajectory) for obj in self.objects)

    @cached_property
    def n_cameras(self) -> int:
        return len(self.camera_array.cameras)

    @cached_property
    def world_points(self) -> WorldPoints:
        """Ground truth 3D points for every object across all frames.

        Static objects contribute a single row per keypoint at STATIC_SYNC_INDEX.
        """
        rows = []
        for scene_obj in self.objects:
            if scene_obj.static:
                world_coords = scene_obj.trajectory.world_points_at_frame(scene_obj.calibration_object, 0)
                for i, kid in enumerate(scene_obj.calibration_object.keypoint_ids):
                    rows.append(
                        {
                            "sync_index": STATIC_SYNC_INDEX,
                            "object_id": scene_obj.object_id,
                            "keypoint_id": int(kid),
                            "x_coord": world_coords[i, 0],
                            "y_coord": world_coords[i, 1],
                            "z_coord": world_coords[i, 2],
                            "frame_time": float("nan"),
                        }
                    )
            else:
                for frame in range(len(scene_obj.trajectory)):
                    world_coords = scene_obj.trajectory.world_points_at_frame(scene_obj.calibration_object, frame)
                    for i, kid in enumerate(scene_obj.calibration_object.keypoint_ids):
                        rows.append(
                            {
                                "sync_index": frame,
                                "object_id": scene_obj.object_id,
                                "keypoint_id": int(kid),
                                "x_coord": world_coords[i, 0],
                                "y_coord": world_coords[i, 1],
                                "z_coord": world_coords[i, 2],
                                "frame_time": frame / 30.0,
                            }
                        )
        return WorldPoints(pd.DataFrame(rows))

    @cached_property
    def image_points_perfect(self) -> ImagePoints:
        """Perfect projections (no noise). Includes obj_loc_x/y/z columns."""
        return self._project_to_cameras(add_noise=False)

    @cached_property
    def image_points_noisy(self) -> ImagePoints:
        """Projections with Gaussian noise added."""
        return self._project_to_cameras(add_noise=True)

    def _project_to_cameras(self, add_noise: bool) -> ImagePoints:
        """Project every object's world points to all cameras.

        Static objects still project a full per-frame trajectory of image
        observations (the collapse to a single world point happens only in
        world_points / triangulation), matching how a real static marker is
        seen anew in every frame.

        Noise is drawn for the full projected block before visibility masking
        so the RNG stream is deterministic regardless of which points are visible.
        """
        rng = np.random.default_rng(self.random_seed) if add_noise else None

        rows = []

        for scene_obj in self.objects:
            n_obj_frames = len(scene_obj.trajectory)
            obj_local = scene_obj.calibration_object.points
            face_normal_local = scene_obj.calibration_object.face_normal

            for frame in range(n_obj_frames):
                world_coords = scene_obj.trajectory.world_points_at_frame(scene_obj.calibration_object, frame)

                # Transform face_normal to world frame if present
                face_normal_world = None
                if face_normal_local is not None:
                    R = scene_obj.trajectory.poses[frame].rotation
                    face_normal_world = R @ face_normal_local

                for cam_id, camera in self.camera_array.cameras.items():
                    if camera.rotation is None or camera.translation is None:
                        continue
                    if camera.matrix is None or camera.distortions is None:
                        continue

                    projected, _ = cv2.projectPoints(
                        world_coords.reshape(-1, 1, 3),
                        camera.rotation,
                        camera.translation,
                        camera.matrix,
                        camera.distortions,
                    )
                    projected = projected.reshape(-1, 2)

                    # Draw noise for all points regardless of masks (RNG stream preservation)
                    if add_noise and rng is not None:
                        noise = rng.normal(0, self.pixel_noise_sigma, projected.shape)
                        projected = projected + noise

                    # Cheirality check: only keep points in front of the camera
                    R_cam = camera.rotation
                    t_cam = camera.translation.reshape(3)
                    p_cam = (R_cam @ world_coords.T).T + t_cam
                    in_front = p_cam[:, 2] > 0

                    # Visibility culling via face_normal
                    if face_normal_world is not None:
                        camera_center = -R_cam.T @ t_cam
                        # For each point, check if camera is on the visible side
                        view_dirs = camera_center - world_coords
                        dots = np.einsum("ij,j->i", view_dirs, face_normal_world)
                        visible = dots > 0
                    else:
                        visible = np.ones(len(world_coords), dtype=bool)

                    w, h = camera.size
                    for i, kid in enumerate(scene_obj.calibration_object.keypoint_ids):
                        if not in_front[i] or not visible[i]:
                            continue

                        x, y = projected[i]

                        if 0 <= x < w and 0 <= y < h:
                            rows.append(
                                {
                                    "sync_index": frame,
                                    "cam_id": cam_id,
                                    "object_id": scene_obj.object_id,
                                    "keypoint_id": int(kid),
                                    "img_loc_x": float(x),
                                    "img_loc_y": float(y),
                                    "obj_loc_x": obj_local[i, 0],
                                    "obj_loc_y": obj_local[i, 1],
                                    "obj_loc_z": obj_local[i, 2],
                                    "frame_time": frame / 30.0,
                                }
                            )

        return ImagePoints(pd.DataFrame(rows))

    @cached_property
    def coverage_matrix(self) -> NDArray[np.int64]:
        """(n_cameras, n_cameras) matrix of shared observation counts."""
        cam_id_to_index = {cam_id: idx for idx, cam_id in enumerate(sorted(self.camera_array.cameras.keys()))}
        return compute_coverage_matrix(self.image_points_noisy, cam_id_to_index)

    def intrinsics_only_cameras(self) -> CameraArray:
        """Return cameras with extrinsics stripped (for pipeline input)."""
        return strip_extrinsics(self.camera_array)

    def apply_filter(self, config: FilterConfig) -> ImagePoints:
        """Apply filters to noisy image points. Returns filtered copy."""
        return config.apply(self.image_points_noisy)
