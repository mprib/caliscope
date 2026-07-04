"""D6.5: Planar/stationary degeneracy characterization.

A stationary board is the degenerate case — globally coplanar world points.
Per-camera PnP from a known planar target is well-posed, so bootstrap
should succeed. The test documents what the pipeline does with this
pathological-but-common capture.
"""

from __future__ import annotations


from caliscope.core.capture_volume import CaptureVolume
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.target_factories import double_sided_charuco_board
from caliscope.synthetic.trajectory import Trajectory
from tests.synthetic.assertions import align_to_ground_truth, pose_error


def _stationary_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """4-camera ring with a stationary (non-moving) board."""
    camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
    calibration_object = double_sided_charuco_board(rows=5, cols=7, square_size=0.05)
    trajectory = Trajectory.stationary(n_frames=10)

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


class TestPlanarDegeneracy:
    def test_bootstrap_succeeds(self) -> None:
        """PnP off a known planar target is well-posed — bootstrap should not crash."""
        scene = _stationary_scene()
        intrinsics_only = scene.intrinsics_only_cameras()
        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        assert cv.camera_array is not None

    def test_optimization_converges(self) -> None:
        """BA with fixed intrinsics has no additional planar degeneracy."""
        scene = _stationary_scene()
        intrinsics_only = scene.intrinsics_only_cameras()
        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize(strict=False)

        assert optimized.reprojection_report.overall_rmse < 5.0

    def test_pose_recovery(self) -> None:
        """Stationary board produces valid poses, possibly with degraded accuracy.

        Tolerances are looser than the moving-board baseline because fewer
        effective constraints exist (all frames are redundant).
        Tolerance: 2.0 deg rotation, 20mm translation — 4x the ring baseline.
        Measured during implementation: worst camera ~0.15 deg, ~1.2mm.
        """
        scene = _stationary_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize(strict=False)

        aligned = align_to_ground_truth(optimized, scene)

        for cam_id in scene.camera_array.cameras:
            err = pose_error(
                aligned.camera_array.cameras[cam_id],
                scene.camera_array.cameras[cam_id],
            )
            assert err.rotation_deg < 2.0, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 2.0 deg"
            assert err.translation_m < 0.020, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 20 mm"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing planar degeneracy...")
    t = TestPlanarDegeneracy()
    t.test_bootstrap_succeeds()
    print("  bootstrap: PASSED")
    t.test_optimization_converges()
    print("  converges: PASSED")
    t.test_pose_recovery()
    print("  pose_recovery: PASSED")
