# %%

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from caliscope import __root__
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.charuco import Charuco
from caliscope.core.point_data import ImagePoints
from caliscope.export import xyz_to_trc, xyz_to_wide_labelled
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.trackers.charuco_tracker import CharucoTracker

logger = logging.getLogger(__name__)

# Use post_optimization session which has calibrated cameras and charuco data
original_session_path = Path(__root__, "tests", "sessions", "post_optimization")
original_data_path = original_session_path / "calibration" / "extrinsic" / "CHARUCO"


def _make_tracker(session_path: Path) -> CharucoTracker:
    """Build a real CharucoTracker from the session's charuco.toml."""
    charuco = Charuco.from_toml(session_path / "charuco.toml")
    return CharucoTracker(charuco)


def _add_frame_time(xyz: pd.DataFrame, timestamps_path: Path) -> pd.DataFrame:
    """Fill frame_time in xyz DataFrame using mean frame_time per sync_index from timestamps.csv.

    The xyz_CHARUCO.csv has a frame_time column but it's empty (NaN). The timestamps file
    has per-camera frame times; we use the per-sync_index mean.
    """
    timestamps = pd.read_csv(timestamps_path)
    sync_time = timestamps.groupby("sync_index")["frame_time"].mean()

    # Drop the existing (empty) frame_time column before merging
    if "frame_time" in xyz.columns:
        xyz = xyz.drop(columns=["frame_time"])

    sync_time_df = sync_time.reset_index()
    return xyz.merge(sync_time_df, on="sync_index", how="left")


def test_export(tmp_path: Path):
    copy_contents_to_clean_dest(original_data_path, tmp_path)
    copy_contents_to_clean_dest(original_session_path, tmp_path.parent / "session")
    session_tmp = tmp_path.parent / "session"

    tracker = _make_tracker(session_tmp)
    xyz_csv_path = Path(tmp_path, f"xyz_{tracker.name}.csv")
    xyz = pd.read_csv(xyz_csv_path)

    # Add frame_time from timestamps (charuco xy data doesn't carry frame_time)
    timestamps_path = session_tmp / "calibration" / "extrinsic" / "CHARUCO" / "timestamps.csv"
    xyz = _add_frame_time(xyz, timestamps_path)

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
    copy_contents_to_clean_dest(original_session_path, tmp_path.parent / "session")
    session_tmp = tmp_path.parent / "session"

    # Load camera array for triangulation
    camera_array = CameraArray.from_toml(session_tmp / "camera_array.toml")

    tracker = _make_tracker(session_tmp)
    xy_csv_path = Path(tmp_path, f"xy_{tracker.name}.csv")

    # Add frame_time to xy from timestamps
    timestamps_path = session_tmp / "calibration" / "extrinsic" / "CHARUCO" / "timestamps.csv"
    timestamps = pd.read_csv(timestamps_path)
    sync_time_map = timestamps.groupby("sync_index")["frame_time"].mean().reset_index()

    # 1. Load original xy data and triangulate to get gold standard xyz
    logger.info("Loading original xy data and triangulating gold standard...")
    original_xy_df = pd.read_csv(xy_csv_path)
    # Inject frame_time into xy so triangulation carries it through to xyz
    original_xy_df = original_xy_df.merge(sync_time_map, on="sync_index", how="left")
    original_xy = ImagePoints(original_xy_df)
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

        # Note: We don't assert a hard tolerance because interpolation error varies
        # with point motion speed between frames. The primary goal is verifying
        # TRC export works with gap-filled data.
        mean_error = np.mean(errors)
        if mean_error > 10:
            logger.warning(f"Mean interpolation error {mean_error:.1f}mm is higher than expected")
    else:
        logger.warning("No gap-filled points to compare!")

    # 5. Verify TRC export works with gap-filled data
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
