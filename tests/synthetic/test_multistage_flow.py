"""D6.4: Multi-stage optimize -> filter -> optimize pipeline.

Verifies the production quality-dialog loop on clean data:
bootstrap -> optimize -> filter_by_percentile_error -> optimize -> align -> compare.

Filtering ordinary Gaussian tails must not corrupt a good solution.
"""

from __future__ import annotations

import cv2
import numpy as np

from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.point_data import STATIC_SYNC_INDEX
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.scene_factories import aruco_scene, default_ring_scene
from caliscope.synthetic.se3_pose import SE3Pose
from caliscope.synthetic.trajectory import Trajectory
from tests.synthetic.assertions import align_to_ground_truth, pose_error


class TestMultistageFlow:
    def test_filter_does_not_degrade_clean_solution(self) -> None:
        """Second optimize after filtering has RMSE <= first optimize."""
        scene = default_ring_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        first = cv.optimize()

        filtered = first.filter_by_percentile_error(percentile=5, scope="per_camera")
        second = filtered.optimize()

        assert second.optimization_status.converged
        assert second.reprojection_report.overall_rmse <= first.reprojection_report.overall_rmse

    def test_poses_within_clean_baseline_after_filter(self) -> None:
        """Post-filter poses match ground truth within default_ring tolerances.

        Tolerances: 0.5 deg rotation, 5mm translation — same as the
        unfiltered default_ring_scene baseline (derived from covariance
        propagation at 0.5 px sigma, 4-cam ring, 20 frames).
        """
        scene = default_ring_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        first = cv.optimize()
        filtered = first.filter_by_percentile_error(percentile=5, scope="per_camera")
        second = filtered.optimize()

        aligned = align_to_ground_truth(second, scene)

        for cam_id in scene.camera_array.cameras:
            err = pose_error(
                aligned.camera_array.cameras[cam_id],
                scene.camera_array.cameras[cam_id],
            )
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"

    def test_static_marker_survives_filtering(self) -> None:
        """Static world rows survive filter -> re-optimize on an aruco scene.

        Regression guard for the filter's static re-attachment logic.
        """
        camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
        markers = {
            0: ArucoMarker(0, 0.1),
            1: ArucoMarker(1, 0.1, static=True),
        }
        marker_set = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)

        mobile_traj = Trajectory.orbital(n_frames=20, radius=0.5)
        static_pose = SE3Pose.from_axis_angle(
            axis=np.array([0.0, 0.0, 1.0]),
            angle_rad=0.0,
            translation=np.array([0.3, -0.2, 0.0]),
        )
        static_traj = Trajectory.stationary(n_frames=20, pose=static_pose)

        scene, constraints = aruco_scene(
            marker_set=marker_set,
            trajectories={0: mobile_traj, 1: static_traj},
            camera_array=camera_array,
        )
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only, constraints=constraints)
        optimized = cv.optimize(use_constraints=True)
        filtered = optimized.filter_by_percentile_error(percentile=5, scope="per_camera")
        reoptimized = filtered.optimize(use_constraints=True)

        static_rows = reoptimized.world_points.df[
            (reoptimized.world_points.df["object_id"] == 1)
            & (reoptimized.world_points.df["sync_index"] == STATIC_SYNC_INDEX)
        ]
        assert len(static_rows) == 4, f"Expected 4 static world rows, got {len(static_rows)}"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing multi-stage flow...")
    t = TestMultistageFlow()
    t.test_filter_does_not_degrade_clean_solution()
    print("  filter_does_not_degrade: PASSED")
    t.test_poses_within_clean_baseline_after_filter()
    print("  poses_within_baseline: PASSED")
    t.test_static_marker_survives_filtering()
    print("  static_marker_survives: PASSED")
