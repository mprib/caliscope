"""Non-planar bootstrap coverage: box target end-to-end and PnP dispatch.

The planar grid and charuco factories never reach the bootstrap's non-planar
(SQPNP) branch. These tests drive a genuinely 3D target (box_target) through
the same bootstrap + optimize path the planar scenes use, and check the
point-count dispatch of compute_camera_to_object_poses_pnp directly.

Tolerances are derived, not tuned. A failure is a finding, not a prompt to
relax the bound.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.bootstrap_pose.pose_network_builder import compute_camera_to_object_poses_pnp
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.point_data import ImagePoints
from caliscope.synthetic.scene_factories import box_target_scene
from tests.synthetic.assertions import align_to_ground_truth, pose_error


class TestNonPlanarBootstrap:
    def test_pose_recovery(self) -> None:
        # 1.0 deg / 10mm. box_target_scene is the default 4-cam ring (radius
        # 2.0, height 0.5), 20 frames, 0.5px sigma — identical geometry to
        # default_ring_scene, which is bounded at 0.5 deg / 5mm by covariance
        # propagation (tests/synthetic/README.md). The box carries 14 points
        # against the grid's 35, so pose std scales by the covariance geometry
        # factor sqrt(N_grid / N_box) = sqrt(35 / 14) ~ 1.58, giving 5mm x 1.58
        # ~ 8mm; rounded up to the README's generic geometry-factor envelope
        # (GEOMETRY_FACTOR 20 x 0.5px = 10mm). Rotation pairs at 1.0 deg,
        # following the suite's convention that a 10mm translation bound carries
        # a 1.0 deg rotation bound (TestOutlierProduction). Measured worst across
        # seeds 42/7/99/123/2024: 0.068 deg / 1.95mm.
        scene = box_target_scene()
        cv = CaptureVolume.bootstrap(scene.image_points_noisy, scene.intrinsics_only_cameras())
        optimized = cv.optimize()
        status = optimized.optimization_status
        assert status is not None and status.converged

        aligned = align_to_ground_truth(optimized, scene)
        for cam_id in scene.camera_array.cameras:
            err = pose_error(aligned.camera_array.cameras[cam_id], scene.camera_array.cameras[cam_id])
            assert err.rotation_deg < 1.0, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 1.0 deg"
            assert err.translation_m < 0.010, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 10 mm"


def _non_planar_group_image_points(num_points: int) -> tuple[ImagePoints, CameraArray]:
    """One camera viewing num_points non-coplanar points at a known pose.

    Projects the points through the camera to get exact img_loc, so a valid
    solve is available when the point count clears the non-planar floor.
    """
    matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    camera = CameraData(cam_id=0, size=(640, 480), matrix=matrix, distortions=np.zeros(5))
    camera_array = CameraArray(cameras={0: camera})

    obj_points = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.02],
            [0.00, 0.10, 0.04],
            [0.10, 0.10, 0.06],
            [0.05, 0.05, 0.10],
            [0.02, 0.08, 0.08],
        ]
    )[:num_points]

    img_points, _ = cv2.projectPoints(obj_points, np.zeros(3), np.array([0.0, 0.0, 1.0]), matrix, np.zeros(5))
    img_points = img_points.reshape(-1, 2)

    df = pd.DataFrame(
        {
            "sync_index": 0,
            "cam_id": 0,
            "object_id": 0,
            "keypoint_id": np.arange(num_points),
            "img_loc_x": img_points[:, 0],
            "img_loc_y": img_points[:, 1],
            "obj_loc_x": obj_points[:, 0],
            "obj_loc_y": obj_points[:, 1],
            "obj_loc_z": obj_points[:, 2],
        }
    )
    return ImagePoints(df), camera_array


class TestNonPlanarDispatch:
    def test_five_points_below_floor_returns_no_pose(self) -> None:
        # A non-planar group of 5 points sits below the non-planar minimum of 6,
        # so no pose is returned for it.
        image_points, camera_array = _non_planar_group_image_points(num_points=5)
        poses = compute_camera_to_object_poses_pnp(image_points, camera_array)
        assert (0, 0, 0) not in poses

    def test_six_points_at_floor_returns_a_pose(self) -> None:
        # Six non-coplanar points clear the floor and SQPNP solves the group.
        image_points, camera_array = _non_planar_group_image_points(num_points=6)
        poses = compute_camera_to_object_poses_pnp(image_points, camera_array)
        assert (0, 0, 0) in poses


if __name__ == "__main__":
    TestNonPlanarBootstrap().test_pose_recovery()
    print("  nonplanar bootstrap pose_recovery: PASSED")
    TestNonPlanarDispatch().test_five_points_below_floor_returns_no_pose()
    TestNonPlanarDispatch().test_six_points_at_floor_returns_a_pose()
    print("  nonplanar dispatch: PASSED")
