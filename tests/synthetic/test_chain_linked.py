"""D6.1: Chain-linked cameras with neighbors-only FOV overlap.

The primary assertion is geometric: the coverage matrix is tridiagonal,
proving only adjacent cameras share observations. The calibration test
verifies the chain topology produces a valid (if looser) solution.
"""

from __future__ import annotations

import numpy as np

from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.alignment import estimate_similarity_transform
from caliscope.synthetic.scene_factories import chain_scene
from tests.synthetic.assertions import align_to_ground_truth, pose_error


class TestChainTopology:
    def test_coverage_matrix_is_tridiagonal(self) -> None:
        """Only adjacent cameras share observations."""
        scene = chain_scene()
        cov = scene.coverage_matrix
        n = cov.shape[0]

        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    assert cov[i, j] == 0, (
                        f"Non-adjacent pair [{i},{j}] has {cov[i, j]} shared observations. "
                        f"Expected tridiagonal structure (neighbors only)."
                    )

    def test_adjacent_pairs_have_observations(self) -> None:
        """Every adjacent pair shares observations."""
        scene = chain_scene()
        cov = scene.coverage_matrix
        n = cov.shape[0]

        for i in range(n - 1):
            assert cov[i, i + 1] > 0, f"Adjacent pair [{i},{i + 1}] has zero shared observations"

    def test_all_cameras_have_observations(self) -> None:
        """Every camera sees the board at some point in the trajectory."""
        scene = chain_scene()
        df = scene.image_points_noisy.df
        for cam_id in scene.camera_array.cameras:
            count = len(df[df["cam_id"] == cam_id])
            assert count > 0, f"Camera {cam_id} has zero observations"


class TestChainCalibration:
    def test_chain_calibrates_within_ceiling(self) -> None:
        """Chain-linked calibration produces a valid solution.

        The optimizer may not formally converge (chain topology is hard),
        but pose errors should stay within empirical ceilings.

        Ceiling: 3x worst single-seed measurement during implementation.
        Measured cam 5 (end of chain): 334mm translation, 3.3 deg rotation.
        Ceiling: 1000mm translation, 10 deg rotation.
        """
        scene = chain_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize(strict=False, max_nfev=5000)

        aligned = align_to_ground_truth(optimized, scene)

        for cam_id in scene.camera_array.cameras:
            err = pose_error(
                aligned.camera_array.cameras[cam_id],
                scene.camera_array.cameras[cam_id],
            )
            assert err.rotation_deg < 10.0, f"cam {cam_id}: rotation {err.rotation_deg:.2f} deg > 10 deg ceiling"
            assert err.translation_m < 1.0, (
                f"cam {cam_id}: translation {err.translation_m * 1000:.1f} mm > 1000 mm ceiling"
            )

    def test_scale_is_metric(self) -> None:
        """Chain reconstruction preserves metric scale (|scale-1| < 0.02)."""
        scene = chain_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize(strict=False, max_nfev=5000)

        from tests.synthetic.assertions import _camera_centers

        opt_centers = _camera_centers(optimized.camera_array)
        gt_centers = _camera_centers(scene.camera_array)
        shared_ids = sorted(set(opt_centers) & set(gt_centers))
        opt_pts = np.array([opt_centers[cid] for cid in shared_ids])
        gt_pts = np.array([gt_centers[cid] for cid in shared_ids])

        sim = estimate_similarity_transform(opt_pts, gt_pts)
        assert abs(sim.scale - 1.0) < 0.02, f"|scale - 1| = {abs(sim.scale - 1.0):.4f}"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing chain topology...")
    t = TestChainTopology()
    t.test_coverage_matrix_is_tridiagonal()
    print("  tridiagonal: PASSED")
    t.test_adjacent_pairs_have_observations()
    print("  adjacent_overlap: PASSED")
    t.test_all_cameras_have_observations()
    print("  all_cameras_visible: PASSED")

    print("\nTesting chain calibration...")
    t2 = TestChainCalibration()
    t2.test_chain_calibrates_within_ceiling()
    print("  within_ceiling: PASSED")
    t2.test_scale_is_metric()
    print("  scale_metric: PASSED")
