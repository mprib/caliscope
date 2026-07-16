"""Gating e2e for thick two-sided charuco calibration (issue #999 spec).

The cross-face linkage is the feature's central claim: the identity split
(back face object_id=1 at z=+t) plus the tie/brace constraints must recover
ground-truth poses through the real production pipeline, in a scene where no
camera ever sees both faces at once.

Tolerances follow the project convention: measured worst over 10 seeds
(rotation 0.035 deg, translation 0.57 mm, world-point RMSE 0.54 mm), set with
~5x margin. The world-point bound doubles as the reflection-basin check: the
distance-only ties cannot distinguish the back face at +t from its reflection
at -t, and a wrong-basin solution would put back-face points ~2t = 12 mm off
ground truth — far outside the 2 mm bound.
"""

from __future__ import annotations

import numpy as np

from caliscope.core.charuco import Charuco
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import ImagePoints
from caliscope.synthetic.scene_factories import two_sided_charuco_scene

from tests.synthetic.production import ProductionRun, run_production_pipeline

ROTATION_TOL_DEG = 0.2
TRANSLATION_TOL_M = 0.003
WORLD_POINT_RMSE_TOL_M = 0.002


def _world_point_errors_m(run: ProductionRun, scene) -> np.ndarray:
    gt = scene.world_points.df.set_index(["sync_index", "object_id", "keypoint_id"])
    opt = run.aligned_volume.world_points.df.set_index(["sync_index", "object_id", "keypoint_id"])
    joined = gt.join(opt, how="inner", lsuffix="_gt")
    return np.sqrt(
        (joined["x_coord"] - joined["x_coord_gt"]) ** 2
        + (joined["y_coord"] - joined["y_coord_gt"]) ** 2
        + (joined["z_coord"] - joined["z_coord_gt"]) ** 2
    ).to_numpy()


def test_scene_shape_matches_tracker_behavior():
    """Each (camera, sync) observes exactly one face — the tracker commits to
    one orientation per frame — and most syncs have both faces triangulable
    (>= 2 cameras each), so the cross-face ties actually fire."""
    scene, _ = two_sided_charuco_scene()
    df = scene.image_points_noisy.df

    faces_per_view = df.groupby(["cam_id", "sync_index"])["object_id"].nunique()
    assert faces_per_view.max() == 1

    cams_per_face = df.groupby(["sync_index", "object_id"])["cam_id"].nunique().unstack(fill_value=0)
    both_triangulable = ((cams_per_face[0] >= 2) & (cams_per_face[1] >= 2)).sum()
    assert both_triangulable >= scene.n_frames // 2


def test_thick_board_pose_recovery_through_production_pipeline():
    """The gating claim: full pipeline (bootstrap, cross-face BA rows, filter,
    re-optimize) recovers ground truth in the correct basin."""
    scene, charuco = two_sided_charuco_scene()
    run = run_production_pipeline(scene, constraints=ConstraintSet.from_charuco(charuco))

    assert run.result.capture_volume.optimization_status is not None
    assert run.result.capture_volume.optimization_status.converged
    assert run.max_rotation_deg < ROTATION_TOL_DEG
    assert run.max_translation_m < TRANSLATION_TOL_M

    errors = _world_point_errors_m(run, scene)
    rmse = float(np.sqrt(np.mean(errors**2)))
    assert rmse < WORLD_POINT_RMSE_TOL_M, f"world-point RMSE {rmse * 1000:.2f}mm (wrong basin would be ~12mm)"


def test_identity_split_beats_fused_treatment():
    """Regression canary on the feature's value: treating the same thick-board
    footage the pre-thickness way (both faces fused as object 0 at z=0) bakes
    in a thickness-sized systematic error. The split-identity path must beat
    it decisively on the same scene and seed."""
    scene, charuco = two_sided_charuco_scene()

    split_run = run_production_pipeline(scene, constraints=ConstraintSet.from_charuco(charuco))

    fused_df = scene.image_points_noisy.df.copy()
    fused_df["object_id"] = 0
    fused_df["obj_loc_z"] = 0.0
    thin = Charuco.from_squares(columns=7, rows=5, square_size_cm=5.0)
    fused_run = run_production_pipeline(
        scene,
        image_points=ImagePoints(fused_df),
        constraints=ConstraintSet.from_charuco(thin),
    )

    # Measured: fused 2.0mm vs split 0.57mm max translation error (6mm board).
    assert split_run.max_translation_m < fused_run.max_translation_m / 2


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent.parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    scene, charuco = two_sided_charuco_scene()
    run = run_production_pipeline(scene, constraints=ConstraintSet.from_charuco(charuco))
    errors = _world_point_errors_m(run, scene)
    print(f"max rotation error: {run.max_rotation_deg:.4f} deg")
    print(f"max translation error: {run.max_translation_m * 1000:.3f} mm")
    print(f"world-point RMSE: {float(np.sqrt(np.mean(errors**2))) * 1000:.3f} mm")
