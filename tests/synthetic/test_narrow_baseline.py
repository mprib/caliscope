"""D6.2: Narrow baseline — characterizes depth uncertainty amplification.

Two cameras with configurable baseline viewing a distant board. Compares
world-point RMSE between narrow (0.1m) and wide (2.0m) baseline.

Measured amplification: 10.4x (theory predicts ~20x, BA halves it).
Narrow: 267mm RMSE, wide: 26mm RMSE.
"""

from __future__ import annotations

import numpy as np

from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.alignment import estimate_similarity_transform
from caliscope.synthetic.scene_factories import narrow_baseline_scene


def _world_point_rmse(spacing: float, seed: int) -> float:
    """World-point RMSE after similarity-transform alignment to ground truth."""
    scene = narrow_baseline_scene(spacing=spacing, random_seed=seed)
    intrinsics_only = scene.intrinsics_only_cameras()

    cv = CaptureVolume.bootstrap(scene.image_points_noisy, intrinsics_only)
    optimized = cv.optimize(strict=False)

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
    aligned = sim.apply(opt_pts)
    return float(np.sqrt(np.mean(np.sum((aligned - gt_pts) ** 2, axis=1))))


class TestNarrowBaseline:
    def test_narrow_error_below_ceiling(self) -> None:
        """Narrow baseline (0.1m) world-point RMSE stays below 500mm.

        Measured mean: 267mm over 10 seeds.
        Ceiling at ~2x to absorb seed variance.
        """
        errors = [_world_point_rmse(spacing=0.1, seed=s) for s in range(5)]
        mean_err = np.mean(errors)

        assert mean_err < 0.500, f"Narrow baseline mean RMSE {mean_err * 1000:.1f} mm > 500 mm ceiling"

    def test_amplification_ratio(self) -> None:
        """Narrow baseline error is at least 5x the wide baseline error.

        Geometric prediction: ~20x (B ratio). Measured: 10.4x over 10 seeds.
        BA reduces the raw amplification by roughly half.
        Conservative floor: 5x.
        """
        seeds = list(range(5))
        narrow_errors = [_world_point_rmse(spacing=0.1, seed=s) for s in seeds]
        wide_errors = [_world_point_rmse(spacing=2.0, seed=s) for s in seeds]

        narrow_mean = np.mean(narrow_errors)
        wide_mean = np.mean(wide_errors)
        ratio = narrow_mean / wide_mean if wide_mean > 0 else float("inf")

        assert ratio >= 5.0, (
            f"Amplification ratio {ratio:.1f}x < 5x. "
            f"Narrow mean={narrow_mean * 1000:.1f} mm, wide mean={wide_mean * 1000:.1f} mm"
        )


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing narrow baseline (5 seeds each, may take a minute)...")
    t = TestNarrowBaseline()
    t.test_narrow_error_below_ceiling()
    print("  ceiling: PASSED")
    t.test_amplification_ratio()
    print("  amplification: PASSED")
