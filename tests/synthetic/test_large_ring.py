"""D6.7: 15-camera stress test.

Smoke test: the sparsity machinery and memory behavior at 10x the
usual observation count. Marked slow — not in the default suite.
"""

from __future__ import annotations

import pytest

from caliscope.core.capture_volume import CaptureVolume
from caliscope.synthetic.scene_factories import large_ring_scene
from tests.synthetic.assertions import align_to_ground_truth, pose_error


@pytest.mark.slow
class TestLargeRing:
    def test_convergence(self) -> None:
        """15-camera ring optimizes and converges."""
        scene = large_ring_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize()

        assert optimized.optimization_status.converged
        assert optimized.reprojection_report.overall_rmse < 2.0

    def test_pose_recovery(self) -> None:
        """All 15 cameras recover within ring-derived tolerances.

        Tolerances: 0.5 deg rotation, 5mm translation — same as 4-camera ring.
        More cameras with full overlap should not degrade accuracy.
        """
        scene = large_ring_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize()

        aligned = align_to_ground_truth(optimized, scene)

        for cam_id in scene.camera_array.cameras:
            err = pose_error(
                aligned.camera_array.cameras[cam_id],
                scene.camera_array.cameras[cam_id],
            )
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing large ring (15 cameras, this may take a minute)...")
    t = TestLargeRing()
    t.test_convergence()
    print("  convergence: PASSED")
    t.test_pose_recovery()
    print("  pose_recovery: PASSED")
