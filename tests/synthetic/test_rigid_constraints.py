"""End-to-end tests for rigid constraints using multi-object synthetic scenes.

Exercises the D2 (static markers), D3 (rigid constraint rows in bundle
adjustment), and D5 (origin setting via align_to_object) acceptance criteria
together, through the synthetic testing framework.
"""

from __future__ import annotations

import cv2
import numpy as np

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import STATIC_SYNC_INDEX
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.multi_object_scene import MultiObjectScene, aruco_scene
from caliscope.synthetic.se3_pose import SE3Pose
from caliscope.synthetic.trajectory import Trajectory


def _make_camera_array() -> CameraArray:
    """4-camera ring, radius 2000mm, height 500mm, facing the origin.

    Markers orbit at height 0 (their local Z-normal points along world Z).
    A camera height offset avoids a degenerate edge-on view of that plane.
    """
    return CameraSynthesizer().add_ring(n=4, radius_mm=2000.0, height_mm=500.0).build()


def _make_aruco_scene() -> tuple[MultiObjectScene, ConstraintSet, CameraArray]:
    """Scene with one mobile (orbiting) marker and one static marker."""
    camera_array = _make_camera_array()
    markers = {
        0: ArucoMarker(0, 0.1),  # mobile, 100mm
        1: ArucoMarker(1, 0.1, static=True),  # static, 100mm
    }
    marker_set = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)

    mobile_traj = Trajectory.orbital(n_frames=20, radius_mm=500.0)
    static_pose = SE3Pose.from_axis_angle(
        axis=np.array([0.0, 0.0, 1.0]),
        angle_rad=0.0,
        translation=np.array([300.0, -200.0, 0.0]),
    )
    static_traj = Trajectory.stationary(n_frames=20, pose=static_pose)

    trajectories = {0: mobile_traj, 1: static_traj}
    scene, constraints = aruco_scene(
        marker_set=marker_set,
        trajectories=trajectories,
        camera_array=camera_array,
        pixel_noise_sigma=0.5,
    )
    return scene, constraints, camera_array


class TestStaticMarkerTriangulation:
    """D2: static markers collapse to a single set of world points."""

    def test_static_marker_produces_single_world_point_set(self) -> None:
        scene, constraints, camera_array = _make_aruco_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        capture_volume = CaptureVolume.bootstrap(
            scene.image_points_noisy,
            intrinsics_only,
            constraints=constraints,
        )

        world_df = capture_volume.world_points.df

        static_rows = world_df[(world_df["object_id"] == 1) & (world_df["sync_index"] == STATIC_SYNC_INDEX)]
        assert len(static_rows) == 4, f"Expected 4 static world rows for marker 1, got {len(static_rows)}"

        # Sentinel sync_index is excluded from min/max
        assert capture_volume.world_points.min_index >= 0
        assert capture_volume.world_points.max_index >= capture_volume.world_points.min_index

        mobile_rows = world_df[world_df["object_id"] == 0]
        assert not mobile_rows.empty, "Expected per-frame world rows for the mobile marker"
        assert set(mobile_rows["sync_index"]) != {STATIC_SYNC_INDEX}


class TestConstrainedVsUnconstrainedOptimization:
    """D3: rigid constraints reduce deformation relative to unconstrained BA."""

    def test_constrained_optimization_has_lower_rigidity_error(self) -> None:
        scene, constraints, camera_array = _make_aruco_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        bootstrapped = CaptureVolume.bootstrap(
            scene.image_points_noisy,
            intrinsics_only,
            constraints=constraints,
        )

        constrained = bootstrapped.optimize(use_constraints=True)
        unconstrained = bootstrapped.optimize(use_constraints=False)

        constrained_rmse = constrained.rigidity_report().rmse_mm
        unconstrained_rmse = unconstrained.rigidity_report().rmse_mm

        assert constrained_rmse < unconstrained_rmse, (
            f"Expected constrained RMSE ({constrained_rmse:.3f}mm) < unconstrained RMSE ({unconstrained_rmse:.3f}mm)"
        )


class TestAlignToStaticObject:
    """D5: align_to_object works with a static marker's collapsed world points."""

    def test_align_to_static_marker_succeeds(self) -> None:
        scene, constraints, camera_array = _make_aruco_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        bootstrapped = CaptureVolume.bootstrap(
            scene.image_points_noisy,
            intrinsics_only,
            constraints=constraints,
        )
        optimized = bootstrapped.optimize(use_constraints=True)

        aligned = optimized.align_to_object(sync_index=None, object_id=1)

        assert aligned is not None
        assert not aligned.world_points.df.empty


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing static marker triangulation...")
    TestStaticMarkerTriangulation().test_static_marker_produces_single_world_point_set()
    print("  PASSED")

    print("\nTesting constrained vs unconstrained optimization...")
    TestConstrainedVsUnconstrainedOptimization().test_constrained_optimization_has_lower_rigidity_error()
    print("  PASSED")

    print("\nTesting align_to_object with static marker...")
    TestAlignToStaticObject().test_align_to_static_marker_succeeds()
    print("  PASSED")
