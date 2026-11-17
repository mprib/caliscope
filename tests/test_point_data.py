# tests/test_point_data.py

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from caliscope.core.calibrate_extrinsics import calibrate_extrinsics
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.exceptions import CalibrationError, CalibrationWarning
from caliscope.synthetic.scene_factories import default_ring_scene
from caliscope.synthetic.synthetic_scene import SyntheticScene


# --- Helper functions for data generation ---
def _get_valid_xy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sync_index": [0, 1, 0, 1],
            "cam_id": [0, 0, 1, 1],
            "object_id": [0, 0, 0, 0],
            "keypoint_id": [10, 10, 10, 10],
            "img_loc_x": [100.5, 102.3, 200.1, 202.8],
            "img_loc_y": [300.2, 301.9, 400.6, 401.3],
            "extra_col": ["a", "b", "c", "d"],
        }
    )


def _get_invalid_xy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sync_index": [0, 1],
            "cam_id": [0, 0],
            "img_loc_x": [100.5, 102.3],
            "img_loc_y": [300.2, 301.9],
        }
    )


def _get_valid_xyz_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sync_index": [0, 1, 2],
            "object_id": [0, 0, 0],
            "keypoint_id": [10, 10, 10],
            "x_coord": [1.1, 1.2, 1.3],
            "y_coord": [2.1, 2.2, 2.3],
            "z_coord": [3.1, 3.2, 3.3],
            "frame_time": [0.0, 0.033, 0.066],
        }
    )


# --- Pytest Fixtures ---
@pytest.fixture
def valid_xy_df() -> pd.DataFrame:
    return _get_valid_xy_df()


@pytest.fixture
def invalid_xy_df() -> pd.DataFrame:
    return _get_invalid_xy_df()


@pytest.fixture
def valid_xyz_df() -> pd.DataFrame:
    return _get_valid_xyz_df()


# --- Test Functions ---
def test_xydata_creation_success(valid_xy_df):
    try:
        xy_data = ImagePoints(valid_xy_df)
        assert isinstance(xy_data, ImagePoints)
        assert not xy_data.df.empty
    except Exception as e:
        pytest.fail(f"XYData creation failed unexpectedly: {e}")


def test_xydata_creation_failure(invalid_xy_df):
    with pytest.raises(ValueError, match="keypoint_id.*not in dataframe"):
        ImagePoints(invalid_xy_df)


def test_xydata_from_csv(valid_xy_df, tmp_path: Path):
    csv_path = tmp_path / "test_xy.csv"
    valid_xy_df.to_csv(csv_path, index=False)
    xy_data = ImagePoints.from_csv(csv_path)
    assert isinstance(xy_data, ImagePoints)


def test_xydata_immutability(valid_xy_df):
    xy_data = ImagePoints(valid_xy_df)
    original_df = xy_data.df
    retrieved_df = xy_data.df
    retrieved_df.iloc[0, 0] = 9999
    pd.testing.assert_frame_equal(original_df, xy_data.df)


def test_xydata_fill_gaps():
    gappy_data = pd.DataFrame(
        {
            "sync_index": [1, 3, 1, 3],
            "cam_id": [0, 0, 1, 1],
            "object_id": [0, 0, 0, 0],
            "keypoint_id": [1, 1, 2, 2],
            "img_loc_x": [10, 30, 100, 300],
            "img_loc_y": [20, 40, 200, 400],
        }
    )
    xy_gappy = ImagePoints(gappy_data)
    xy_filled = xy_gappy.fill_gaps(max_gap_size=3)

    expected_data = (
        pd.DataFrame(
            {
                "sync_index": [1, 2, 3, 1, 2, 3],
                "cam_id": [0, 0, 0, 1, 1, 1],
                "object_id": [0, 0, 0, 0, 0, 0],
                "keypoint_id": [1, 1, 1, 2, 2, 2],
                "img_loc_x": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
                "img_loc_y": [20.0, 30.0, 40.0, 200.0, 300.0, 400.0],
            }
        )
        .sort_values(["cam_id", "object_id", "keypoint_id", "sync_index"])
        .reset_index(drop=True)
    )
    result_df = (
        xy_filled.df[["sync_index", "cam_id", "object_id", "keypoint_id", "img_loc_x", "img_loc_y"]]
        .sort_values(["cam_id", "object_id", "keypoint_id", "sync_index"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(expected_data, result_df)


def test_untracked_points_have_no_track_id_after_csv(valid_xy_df, tmp_path: Path):
    csv_path = tmp_path / "xy.csv"
    ImagePoints(valid_xy_df).to_csv(csv_path)

    assert "track_id" not in ImagePoints.from_csv(csv_path).df.columns


def test_tracked_points_keep_int64_track_id_after_csv(valid_xy_df, tmp_path: Path):
    valid_xy_df["track_id"] = [4, 4, 7, 7]
    csv_path = tmp_path / "xy.csv"
    ImagePoints(valid_xy_df).to_csv(csv_path)

    loaded = ImagePoints.from_csv(csv_path).df
    assert loaded["track_id"].dtype == "int64"
    assert loaded["track_id"].tolist() == [4, 4, 7, 7]


def test_track_id_with_missing_value_rejected(valid_xy_df):
    valid_xy_df["track_id"] = [4, None, 7, 7]

    with pytest.raises(ValueError, match="track_id"):
        ImagePoints(valid_xy_df)


def test_negative_object_id_shared_by_two_cameras_rejected(valid_xy_df):
    valid_xy_df["object_id"] = -1

    with pytest.raises(ValueError, match="span more than one cam_id"):
        ImagePoints(valid_xy_df)


# --- Mixed-ID rule ---

ScenePointsConsumer = Callable[[ImagePoints, SyntheticScene], object]

CONSUMERS: dict[str, ScenePointsConsumer] = {
    "triangulate": lambda points, scene: points.triangulate(scene.camera_array),
    "bootstrap": lambda points, scene: CaptureVolume.bootstrap(points, scene.intrinsics_only_cameras()),
    "calibrate_extrinsics": lambda points, scene: calibrate_extrinsics(
        points, scene.intrinsics_only_cameras(), None, refine_intrinsics=False
    ),
}


@pytest.fixture(scope="module")
def ring_scene() -> SyntheticScene:
    return default_ring_scene()


@pytest.mark.parametrize("consumer", list(CONSUMERS.values()), ids=list(CONSUMERS))
def test_entirely_unlinked_points_raise(consumer: ScenePointsConsumer, ring_scene: SyntheticScene):
    df = ring_scene.image_points_noisy.df
    df["object_id"] = -(df["cam_id"] + 1)
    df["track_id"] = df["cam_id"]

    with pytest.raises(CalibrationError, match="associate_tracks"):
        consumer(ImagePoints(df), ring_scene)


@pytest.mark.parametrize("consumer", list(CONSUMERS.values()), ids=list(CONSUMERS))
def test_mixed_points_warn_once_with_the_unlinked_count(consumer: ScenePointsConsumer, ring_scene: SyntheticScene):
    linked = ring_scene.image_points_noisy.df
    linked["track_id"] = linked["cam_id"]
    unlinked = linked[linked["cam_id"] == linked["cam_id"].min()].copy()
    unlinked["object_id"] = -1
    unlinked["track_id"] = 99
    points = ImagePoints(pd.concat([linked, unlinked], ignore_index=True))

    with pytest.warns(CalibrationWarning) as record:
        consumer(points, ring_scene)

    calibration_warnings = [w for w in record if issubclass(w.category, CalibrationWarning)]
    assert len(calibration_warnings) == 1
    assert f"dropped {len(unlinked)} of {len(points.df)}" in str(calibration_warnings[0].message)


# --- WorldPoints Tests ---
def test_worldpoints_from_csv(valid_xyz_df: pd.DataFrame, tmp_path: Path):
    """Test loading WorldPoints from CSV file."""
    csv_path = tmp_path / "test_xyz.csv"
    valid_xyz_df.to_csv(csv_path, index=False)
    xyz_data = WorldPoints.from_csv(csv_path)
    assert isinstance(xyz_data, WorldPoints)
    assert xyz_data.min_index == 0
    assert xyz_data.max_index == 2
    assert len(xyz_data.df) == 3


def test_worldpoints_from_csv_with_frame_time(tmp_path: Path):
    """Test loading WorldPoints with optional frame_time column."""
    df = pd.DataFrame(
        {
            "sync_index": [0, 1],
            "object_id": [0, 0],
            "keypoint_id": [1, 1],
            "x_coord": [1.0, 1.1],
            "y_coord": [2.0, 2.1],
            "z_coord": [3.0, 3.1],
            "frame_time": [0.0, 0.033],
        }
    )
    csv_path = tmp_path / "test_xyz_with_time.csv"
    df.to_csv(csv_path, index=False)
    xyz_data = WorldPoints.from_csv(csv_path)
    assert "frame_time" in xyz_data.df.columns


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print("--- Running test: test_xydata_creation_success ---")
        test_xydata_creation_success(_get_valid_xy_df())
        print("\n--- Running test: test_xydata_creation_failure ---")
        test_xydata_creation_failure(_get_invalid_xy_df())
        print("\n--- Running test: test_xydata_from_csv ---")
        test_xydata_from_csv(_get_valid_xy_df(), temp_path)
        print("\n--- Running test: test_xydata_immutability ---")
        test_xydata_immutability(_get_valid_xy_df())
        print("\n--- Running test: test_xydata_fill_gaps ---")
        test_xydata_fill_gaps()
        print("\n--- All tests passed in debug mode ---")
