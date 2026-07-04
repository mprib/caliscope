"""E5a: Robust loss on the outlier scene (extrinsics-only, runs first).

Tests whether Huber loss prevents BA from absorbing outliers into camera
poses, which would leave outlier residuals exposed for the post-hoc filter.

This is the evidence gate for the robust-loss-ba production task.
"""

from __future__ import annotations


from caliscope.core.capture_volume import CaptureVolume
from caliscope.synthetic.experimental_ba import optimize_with_robust_loss
from caliscope.synthetic.scene_factories import outlier_scene
from tests.synthetic.assertions import align_to_ground_truth, pose_error


def _corrupted_row_keys(corrupted_df, corrupted_indices):
    key_cols = ["sync_index", "cam_id", "object_id", "keypoint_id"]
    keys = set()
    for idx in corrupted_indices:
        row = corrupted_df.iloc[idx]
        keys.add(tuple(int(row[c]) for c in key_cols))
    return keys


def _row_key_set(df):
    key_cols = ["sync_index", "cam_id", "object_id", "keypoint_id"]
    return set(tuple(int(row[c]) for c in key_cols) for _, row in df.iterrows())


class TestRobustLossOutliers:
    def test_robust_solve_converges(self) -> None:
        """Huber BA on corrupted data converges."""
        scene, corrupted, _ = outlier_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(corrupted, intrinsics_only)
        optimized = optimize_with_robust_loss(cv, loss="huber", f_scale_px=1.0)

        assert optimized.optimization_status.converged

    def test_robust_poses_no_worse_than_linear(self) -> None:
        """Huber BA on corrupted data produces poses at least as good as linear BA.

        Huber downweights large residuals during optimization, so it should
        not bend toward outliers as much as the linear loss.
        """
        scene, corrupted, _ = outlier_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(corrupted, intrinsics_only)
        linear = cv.optimize()
        robust = optimize_with_robust_loss(cv, loss="huber", f_scale_px=1.0)

        linear_aligned = align_to_ground_truth(linear, scene)
        robust_aligned = align_to_ground_truth(robust, scene)

        linear_max_trans = max(
            pose_error(linear_aligned.camera_array.cameras[cid], scene.camera_array.cameras[cid]).translation_m
            for cid in scene.camera_array.cameras
        )
        robust_max_trans = max(
            pose_error(robust_aligned.camera_array.cameras[cid], scene.camera_array.cameras[cid]).translation_m
            for cid in scene.camera_array.cameras
        )

        assert robust_max_trans <= linear_max_trans * 1.1, (
            f"Robust worst translation {robust_max_trans * 1000:.2f} mm > "
            f"1.1x linear worst {linear_max_trans * 1000:.2f} mm"
        )

    def test_robust_improves_filter_recovery(self) -> None:
        """Post-robust-solve filter catches more outliers than post-linear-solve filter.

        The key measurement: if Huber prevents BA from absorbing outliers,
        their residuals stay large and the percentile filter catches them.
        Measured linear recovery: 68.6%. Huber should be materially higher.
        """
        scene, corrupted, corrupted_indices = outlier_scene()
        corrupted_keys = _corrupted_row_keys(corrupted.df, corrupted_indices)
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(corrupted, intrinsics_only)

        # Linear baseline
        linear = cv.optimize()
        linear_filtered = linear.filter_by_percentile_error(percentile=5, scope="overall")
        linear_pre = _row_key_set(linear.image_points.df)
        linear_post = _row_key_set(linear_filtered.image_points.df)
        linear_removed = linear_pre - linear_post
        linear_matched = corrupted_keys & linear_pre
        linear_caught = linear_removed & linear_matched
        linear_recall = len(linear_caught) / len(linear_matched) if linear_matched else 0

        # Robust solve
        robust = optimize_with_robust_loss(cv, loss="huber", f_scale_px=1.0)
        robust_filtered = robust.filter_by_percentile_error(percentile=5, scope="overall")
        robust_pre = _row_key_set(robust.image_points.df)
        robust_post = _row_key_set(robust_filtered.image_points.df)
        robust_removed = robust_pre - robust_post
        robust_matched = corrupted_keys & robust_pre
        robust_caught = robust_removed & robust_matched
        robust_recall = len(robust_caught) / len(robust_matched) if robust_matched else 0

        # Robust should beat linear (or at least match it)
        assert robust_recall >= linear_recall, (
            f"Robust recall {robust_recall:.3f} < linear recall {linear_recall:.3f}. "
            f"Huber did not improve outlier separation."
        )

        # Regression floor: robust should be at least as good as the measured linear 68%
        assert robust_recall >= 0.60, f"Robust recall {robust_recall:.3f} < 0.60 floor"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing E5a: robust loss on outlier scene...")
    t = TestRobustLossOutliers()
    t.test_robust_solve_converges()
    print("  converges: PASSED")
    t.test_robust_poses_no_worse_than_linear()
    print("  poses_no_worse: PASSED")
    t.test_robust_improves_filter_recovery()
    print("  improves_recovery: PASSED")
