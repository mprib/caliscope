"""Synthetic scene with multiple independently-moving calibration objects."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
import pandas as pd

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.aruco_marker import ArucoMarkerSet
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import STATIC_SYNC_INDEX, ImagePoints, WorldPoints
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import strip_extrinsics
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
class MultiObjectScene:
    """Complete synthetic scenario with several independently-posed objects.

    Mirrors SyntheticScene but supports multiple SceneObjects sharing a
    camera rig, enabling tests of rigid constraints across objects.
    """

    camera_array: CameraArray
    objects: tuple[SceneObject, ...]
    pixel_noise_sigma: float = 0.5
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate scene configuration."""
        if self.pixel_noise_sigma < 0:
            raise ValueError(f"pixel_noise_sigma must be >= 0, got {self.pixel_noise_sigma}")

        unposed = self.camera_array.unposed_cameras
        if unposed:
            raise ValueError(f"All cameras must have extrinsics. Unposed: {list(unposed.keys())}")

        if not self.objects:
            raise ValueError("At least one SceneObject required")

    @cached_property
    def n_frames(self) -> int:
        """Number of frames spanned by the longest object trajectory."""
        return max(len(obj.trajectory) for obj in self.objects)

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
    def image_points_noisy(self) -> ImagePoints:
        """Projections with Gaussian noise added."""
        return self._project_to_cameras(add_noise=True)

    @cached_property
    def image_points_perfect(self) -> ImagePoints:
        """Perfect projections (no noise)."""
        return self._project_to_cameras(add_noise=False)

    def _project_to_cameras(self, add_noise: bool) -> ImagePoints:
        """Project every object's world points to all cameras.

        Static objects still project a full per-frame trajectory of image
        observations (the collapse to a single world point happens only in
        world_points / triangulation), matching how a real static marker is
        seen anew in every frame.
        """
        rng = np.random.default_rng(self.random_seed) if add_noise else None

        rows = []

        for scene_obj in self.objects:
            n_obj_frames = len(scene_obj.trajectory)
            obj_local = scene_obj.calibration_object.points

            for frame in range(n_obj_frames):
                world_coords = scene_obj.trajectory.world_points_at_frame(scene_obj.calibration_object, frame)

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

                    if add_noise and rng is not None:
                        noise = rng.normal(0, self.pixel_noise_sigma, projected.shape)
                        projected = projected + noise

                    w, h = camera.size
                    for i, kid in enumerate(scene_obj.calibration_object.keypoint_ids):
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

    def intrinsics_only_cameras(self) -> CameraArray:
        """Return cameras with extrinsics stripped (for pipeline input)."""
        return strip_extrinsics(self.camera_array)


def aruco_scene(
    marker_set: ArucoMarkerSet,
    trajectories: dict[int, Trajectory],
    camera_array: CameraArray,
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> tuple[MultiObjectScene, ConstraintSet]:
    """Build a MultiObjectScene and matching ConstraintSet from an ArucoMarkerSet.

    Markers without an entry in `trajectories` are skipped. Marker corners
    (meters) are scaled by 1000 to match the synthetic framework's millimeter
    convention; the returned ConstraintSet uses unit_scale=1000.0 accordingly.
    """
    objects = []
    for marker_id, marker in marker_set.markers.items():
        if marker_id not in trajectories:
            continue
        cal_obj = CalibrationObject.from_points(marker.corners * 1000.0)
        objects.append(
            SceneObject(
                object_id=marker_id,
                calibration_object=cal_obj,
                trajectory=trajectories[marker_id],
                static=marker.static,
            )
        )

    scene = MultiObjectScene(
        camera_array=camera_array,
        objects=tuple(objects),
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )
    constraints = ConstraintSet.from_marker_set(marker_set, unit_scale=1000.0)
    return scene, constraints
