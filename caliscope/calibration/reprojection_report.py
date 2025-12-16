from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ReprojectionReport:
    """Comprehensive snapshot of reprojection error metrics."""

    # Aggregate metrics
    overall_rmse: float
    by_camera: dict[int, float]  # port -> rmse
    by_point_id: dict[int, float]  # point_id -> rmse

    # Unmatched observation tracking
    n_unmatched_observations: int
    unmatched_rate: float
    unmatched_by_camera: dict[int, int]

    # Raw matched errors for detailed analysis
    raw_errors: pd.DataFrame  # columns: sync_index, port, point_id, error_x, error_y, euclidean_error

    # Quality metadata
    n_observations_matched: int
    n_observations_total: int
    n_cameras: int
    n_points: int
