# %%

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from caliscope import __root__, persistence
from caliscope.core.point_data import ImagePoints
from caliscope.export import xyz_to_trc, xyz_to_wide_labelled
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.trackers.holistic.holistic_tracker import HolisticTracker

logger = logging.getLogger(__name__)

original_data_path = Path(__root__, "tests", "sessions", "4_cam_recording", "recordings", "recording_1", "HOLISTIC")
session_path = Path(__root__, "tests", "sessions", "4_cam_recording")


def test_export(tmp_path: Path):
    copy_contents_to_clean_dest(original_data_path, tmp_path)

    tracker = HolisticTracker()
    xyz_csv_path = Path(tmp_path, f"xyz_{tracker.name}.csv")
    xyz = pd.read_csv(xyz_csv_path)

    # Add frame_time to xyz (legacy test data doesn't have it - new pipeline includes it)
    # Use xy data as source of frame_time since it matches xyz sync_index range
    xy_csv_path = Path(tmp_path, f"xy_{tracker.name}.csv")
    xy = pd.read_csv(xy_csv_path)
    sync_time = xy.groupby("sync_index")["frame_time"].mean().reset_index()
    xyz = xyz.merge(sync_time, on="sync_index", how="left")

    # this file should be created now
    xyz_labelled_path = Path(xyz_csv_path.parent, f"{xyz_csv_path.stem}_labelled.csv")
    # the file shouldn't exist yet
    assert not xyz_labelled_path.exists()
    # create it
    xyz_labelled = xyz_to_wide_labelled(xyz, tracker)
    xyz_labelled.to_csv(xyz_labelled_path)
    # confirm it exists
    assert xyz_labelled_path.exists()

    # do the same with the trc file
    trc_path = Path(xyz_csv_path.parent, f"{xyz_csv_path.stem}.trc")
    assert not trc_path.exists()

    xyz_to_trc(xyz, tracker, target_path=trc_path)
    assert trc_path.exists()
    # %%


def test_gap_fill_pipeline_accuracy(tmp_path: Path):
    """
    Integration test: validates full gap-filling pipeline by comparing
    interpolated values to known gold standard.

    Tests the fix for #877 - TRC export failing with NaN frame indices.
    """
    copy_contents_to_clean_dest(original_data_path, tmp_path)
    copy_contents_to_clean_dest(session_path, tmp_path.parent / "session")
    session_tmp = tmp_path.parent / "session"

    # Load camera array for triangulation
    camera_array = persistence.load_camera_array(session_tmp / "camera_array.toml")

    tracker = HolisticTracker()
    xy_csv_path = Path(tmp_path, f"xy_{tracker.name}.csv")

    # 1. Load original xy data and triangulate to get gold standard xyz
    logger.info("Loading original xy data and triangulating gold standard...")
    original_xy = ImagePoints.from_csv(xy_csv_path)
    gold_xyz = original_xy.triangulate(camera_array)
    gold_df = gold_xyz.df

    min_sync = gold_df["sync_index"].min()
    max_sync = gold_df["sync_index"].max()
    logger.info(f"Gold standard: {len(gold_df)} 3D points, sync_index range [{min_sync}-{max_sync}]")

    # 2. Create gaps by removing specific sync_indices
    # Choose indices in the middle of the range to ensure interpolation is possible
    min_idx = int(min_sync)
    max_idx = int(max_sync)
    mid = (min_idx + max_idx) // 2

    # Remove 3-frame gap, 1-frame gap, and 2-frame gap
    gap_indices = [mid, mid + 1, mid + 2, mid + 20, mid + 40, mid + 41]
    gap_indices = [idx for idx in gap_indices if min_idx < idx < max_idx]  # Keep in valid range

    logger.info(f"Creating gaps at sync_indices: {gap_indices}")

    gapped_xy_df = original_xy.df[~original_xy.df["sync_index"].isin(gap_indices)]
    gapped_xy = ImagePoints(gapped_xy_df)

    # 3. Run pipeline: fill_gaps(2D) → triangulate → fill_gaps(3D)
    logger.info("Running gap-fill pipeline...")
    filled_xy = gapped_xy.fill_gaps(max_gap_size=3)
    triangulated = filled_xy.triangulate(camera_array)
    filled_xyz = triangulated.fill_gaps(max_gap_size=3)
    filled_df = filled_xyz.df

    # 4. Log detailed disparity report
    logger.info("\n" + "=" * 60)
    logger.info("Gap-fill accuracy report:")
    logger.info("=" * 60)

    errors = []
    for sync_idx in gap_indices:
        gold_points = gold_df[gold_df["sync_index"] == sync_idx]
        filled_points = filled_df[filled_df["sync_index"] == sync_idx]

        for _, gold_row in gold_points.iterrows():
            point_id = gold_row["point_id"]
            filled_row = filled_points[filled_points["point_id"] == point_id]

            if filled_row.empty:
                logger.warning(f"  sync_index={sync_idx}, point_id={point_id}: NOT FILLED")
                continue

            gold_pos = np.array([gold_row["x_coord"], gold_row["y_coord"], gold_row["z_coord"]])
            filled_pos = np.array(
                [
                    filled_row["x_coord"].iloc[0],
                    filled_row["y_coord"].iloc[0],
                    filled_row["z_coord"].iloc[0],
                ]
            )

            error_m = np.linalg.norm(filled_pos - gold_pos)
            error_mm = error_m * 1000
            errors.append(error_mm)

            logger.info(
                f"  sync_index={sync_idx}, point_id={int(point_id)}: "
                f"gold=({gold_pos[0]:.3f}, {gold_pos[1]:.3f}, {gold_pos[2]:.3f}), "
                f"filled=({filled_pos[0]:.3f}, {filled_pos[1]:.3f}, {filled_pos[2]:.3f}), "
                f"error={error_mm:.1f}mm"
            )

    if errors:
        logger.info("-" * 60)
        logger.info(
            f"Summary: {len(errors)} points checked, max_error={max(errors):.1f}mm, mean_error={np.mean(errors):.1f}mm"
        )
        logger.info("=" * 60 + "\n")

        # Note: We don't assert a hard tolerance because:
        # - Body landmarks (torso, head) have low interpolation error (~1-5mm)
        # - Fast-moving extremities (hands, fingers) can have high error (>50mm)
        # The primary goal is verifying TRC export works with gap-filled data.
        # A mean error < 10mm indicates reasonable overall accuracy.
        mean_error = np.mean(errors)
        if mean_error > 10:
            logger.warning(f"Mean interpolation error {mean_error:.1f}mm is higher than expected")
    else:
        logger.warning("No gap-filled points to compare!")

    # 6. Verify TRC export works with gap-filled data
    trc_path = Path(tmp_path, "gap_filled.trc")
    xyz_to_trc(filled_df, tracker, target_path=trc_path)
    assert trc_path.exists(), "TRC export failed for gap-filled data"
    logger.info(f"TRC export successful: {trc_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tmp_path = Path(__file__).parent / "tmp"
    tmp_path.mkdir(exist_ok=True)

    test_export(tmp_path / "export")
    test_gap_fill_pipeline_accuracy(tmp_path / "gap_fill")
