"""D6.6: Scale-gauge characterization.

Verifies that:
1. Raw (unaligned) pose comparison fails — the optimized frame is arbitrary.
2. The reconstruction is metric from the start (PnP uses known obj_loc).
3. After similarity transform, poses match ground truth.
"""

from __future__ import annotations


from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.alignment import estimate_similarity_transform, apply_similarity_transform
from caliscope.synthetic.scene_factories import default_ring_scene
from tests.synthetic.assertions import pose_error


class TestAlignmentGauge:
    def test_unaligned_comparison_is_meaningless(self) -> None:
        """At least one camera's raw translation error exceeds 10x the aligned tolerance.

        The optimized world lives in the best-scoring camera's frame, not the
        ground truth frame. Direct comparison without alignment is meaningless.
        """
        scene = default_ring_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize()

        aligned_tol_m = 0.005
        threshold = 10 * aligned_tol_m

        max_err = 0.0
        for cam_id in scene.camera_array.cameras:
            err = pose_error(
                optimized.camera_array.cameras[cam_id],
                scene.camera_array.cameras[cam_id],
            )
            max_err = max(max_err, err.translation_m)

        assert max_err > threshold, (
            f"Raw comparison worst error {max_err * 1000:.1f} mm <= {threshold * 1000:.1f} mm threshold. "
            f"Expected unaligned comparison to fail dramatically."
        )

    def test_scale_is_metric_without_alignment(self) -> None:
        """Similarity transform from optimized to ground truth has |scale-1| < 0.01.

        PnP bootstrap uses known board geometry (obj_loc in meters), so the
        reconstruction is metric from the start. Alignment fixes the frame,
        not the scale.
        """
        scene = default_ring_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize()

        gt_df = scene.world_points.df
        opt_df = optimized.world_points.df
        merged = gt_df.merge(
            opt_df,
            on=["sync_index", "object_id", "keypoint_id"],
            suffixes=("_gt", "_opt"),
            how="inner",
        )
        gt_pts = merged[["x_coord_gt", "y_coord_gt", "z_coord_gt"]].to_numpy()
        opt_pts = merged[["x_coord_opt", "y_coord_opt", "z_coord_opt"]].to_numpy()

        sim = estimate_similarity_transform(opt_pts, gt_pts)

        assert abs(sim.scale - 1.0) < 0.01, (
            f"|scale - 1| = {abs(sim.scale - 1.0):.4f} >= 0.01. PnP should produce metric reconstruction."
        )

    def test_aligned_poses_match_ground_truth(self) -> None:
        """Post-alignment pose errors within default_ring tolerances.

        Tolerances: 0.5 deg rotation, 5mm translation.
        """
        scene = default_ring_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
        optimized = cv.optimize()

        gt_df = scene.world_points.df
        opt_df = optimized.world_points.df
        merged = gt_df.merge(
            opt_df,
            on=["sync_index", "object_id", "keypoint_id"],
            suffixes=("_gt", "_opt"),
            how="inner",
        )
        gt_pts = merged[["x_coord_gt", "y_coord_gt", "z_coord_gt"]].to_numpy()
        opt_pts = merged[["x_coord_opt", "y_coord_opt", "z_coord_opt"]].to_numpy()

        sim = estimate_similarity_transform(opt_pts, gt_pts)
        aligned_cameras, _ = apply_similarity_transform(optimized.camera_array, optimized.world_points, sim)

        for cam_id in scene.camera_array.cameras:
            err = pose_error(
                aligned_cameras.cameras[cam_id],
                scene.camera_array.cameras[cam_id],
            )
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing alignment gauge...")
    t = TestAlignmentGauge()
    t.test_unaligned_comparison_is_meaningless()
    print("  unaligned_meaningless: PASSED")
    t.test_scale_is_metric_without_alignment()
    print("  scale_is_metric: PASSED")
    t.test_aligned_poses_match_ground_truth()
    print("  aligned_poses_match: PASSED")
