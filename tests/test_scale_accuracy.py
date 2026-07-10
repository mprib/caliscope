import numpy as np
import pytest
from scipy.spatial.distance import pdist

from caliscope.core.point_data import STATIC_SYNC_INDEX
from caliscope.core.scale_accuracy import (
    FrameScaleError,
    VolumetricScaleReport,
    compute_frame_scale_error,
)


def make_frame_error(
    sync_index: int,
    object_id: int,
    sum_squared_errors_m2: float,
    sum_squared_relative_errors: float,
    n_distance_pairs: int = 1,
) -> FrameScaleError:
    return FrameScaleError(
        sync_index=sync_index,
        object_id=object_id,
        distance_rmse_mm=0.0,
        distance_mean_signed_error_mm=0.0,
        distance_max_error_mm=0.0,
        n_corners=3,
        n_distance_pairs=n_distance_pairs,
        n_cameras_contributing=2,
        sum_squared_errors_m2=sum_squared_errors_m2,
        sum_squared_relative_errors=sum_squared_relative_errors,
        centroid=(0.0, 0.0, 0.0),
    )


def test_sum_squared_relative_errors_matches_normalization_identity():
    """sum_squared_relative_errors == sum_squared_errors_m2 / D_ref^2, D_ref = max nominal pairwise distance."""
    object_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    world_points = object_points + np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]])

    fe = compute_frame_scale_error(
        world_points=world_points,
        object_points=object_points,
        sync_index=0,
        object_id=1,
        n_cameras_contributing=2,
    )

    D_ref = float(np.max(pdist(object_points)))
    expected_rel = fe.sum_squared_errors_m2 / (D_ref**2)
    assert fe.sum_squared_relative_errors == pytest.approx(expected_rel)


def test_pooled_relative_rmse_pct_pools_across_frames():
    fe1 = make_frame_error(0, 1, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.01, n_distance_pairs=2)
    fe2 = make_frame_error(1, 1, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.03, n_distance_pairs=2)
    report = VolumetricScaleReport(frame_errors=(fe1, fe2))

    expected = np.sqrt((0.01 + 0.03) / (2 + 2)) * 100
    assert report.pooled_relative_rmse_pct == pytest.approx(expected)


def test_pooled_relative_rmse_pct_empty_report():
    report = VolumetricScaleReport(frame_errors=())
    assert report.pooled_relative_rmse_pct == 0.0


def test_per_frame_relative_rmse_pct_excludes_static_sync_index():
    fe_moving = make_frame_error(5, 1, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.04, n_distance_pairs=1)
    fe_static = make_frame_error(
        STATIC_SYNC_INDEX, 2, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.09, n_distance_pairs=1
    )
    report = VolumetricScaleReport(frame_errors=(fe_moving, fe_static))

    result = report.per_frame_relative_rmse_pct
    assert 5 in result
    assert STATIC_SYNC_INDEX not in result
    assert result[5] == pytest.approx(np.sqrt(0.04 / 1) * 100)


def test_per_object_relative_rmse_pct_includes_static_objects():
    fe_moving = make_frame_error(0, 1, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.04, n_distance_pairs=1)
    fe_static = make_frame_error(
        STATIC_SYNC_INDEX, 2, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.09, n_distance_pairs=1
    )
    report = VolumetricScaleReport(frame_errors=(fe_moving, fe_static))

    result = report.per_object_relative_rmse_pct
    assert result.keys() == {1, 2}
    assert result[1] == pytest.approx(np.sqrt(0.04 / 1) * 100)
    assert result[2] == pytest.approx(np.sqrt(0.09 / 1) * 100)


def test_split_relative_rmse_pct_partitions_by_static_object_ids():
    fe_moving = make_frame_error(0, 1, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.04, n_distance_pairs=1)
    fe_static = make_frame_error(
        STATIC_SYNC_INDEX, 2, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.09, n_distance_pairs=1
    )
    report = VolumetricScaleReport(frame_errors=(fe_moving, fe_static), static_object_ids=frozenset({2}))

    moving_pct, static_pct = report.split_relative_rmse_pct
    assert moving_pct == pytest.approx(np.sqrt(0.04 / 1) * 100)
    assert static_pct == pytest.approx(np.sqrt(0.09 / 1) * 100)


def test_split_relative_rmse_pct_none_for_empty_side():
    fe_moving = make_frame_error(0, 1, sum_squared_errors_m2=0.0, sum_squared_relative_errors=0.04, n_distance_pairs=1)
    report = VolumetricScaleReport(frame_errors=(fe_moving,), static_object_ids=frozenset({99}))

    moving_pct, static_pct = report.split_relative_rmse_pct
    assert moving_pct == pytest.approx(np.sqrt(0.04 / 1) * 100)
    assert static_pct is None
