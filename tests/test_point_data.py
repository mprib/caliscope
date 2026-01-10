# tests/test_point_data.py

from pathlib import Path

import pandas as pd
import pandera
import pytest

from caliscope.core.point_data import ImagePoints


# --- Helper functions for data generation ---
def _get_valid_xy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sync_index": [0, 1, 0, 1],
            "port": [0, 0, 1, 1],
            "point_id": [10, 10, 10, 10],
            "img_loc_x": [100.5, 102.3, 200.1, 202.8],
            "img_loc_y": [300.2, 301.9, 400.6, 401.3],
            "extra_col": ["a", "b", "c", "d"],
        }
    )


def _get_invalid_xy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sync_index": [0, 1],
            "port": [0, 0],
            "img_loc_x": [100.5, 102.3],
            "img_loc_y": [300.2, 301.9],
        }
    )


def _get_valid_xyz_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sync_index": [0, 1, 2],
            "point_id": [10, 10, 10],
            "x_coord": [1.1, 1.2, 1.3],
            "y_coord": [2.1, 2.2, 2.3],
            "z_coord": [3.1, 3.2, 3.3],
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
    # pandera.errors exists at runtime but type stubs don't export it
    with pytest.raises(pandera.errors.SchemaError) as excinfo:  # type: ignore[attr-defined]
        ImagePoints(invalid_xy_df)
    assert "column 'point_id' not in dataframe" in str(excinfo.value)


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
            "port": [0, 0, 1, 1],
            "point_id": [1, 1, 2, 2],
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
                "port": [0, 0, 0, 1, 1, 1],
                "point_id": [1, 1, 1, 2, 2, 2],
                "img_loc_x": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
                "img_loc_y": [20.0, 30.0, 40.0, 200.0, 300.0, 400.0],
            }
        )
        .sort_values(["port", "point_id", "sync_index"])
        .reset_index(drop=True)
    )
    result_df = (
        xy_filled.df[["sync_index", "port", "point_id", "img_loc_x", "img_loc_y"]]
        .sort_values(["port", "point_id", "sync_index"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(expected_data, result_df)


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
