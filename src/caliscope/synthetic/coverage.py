"""Coverage matrix computation for camera observation analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from caliscope.core.point_data import ImagePoints


def compute_coverage_matrix(image_points: ImagePoints, n_cameras: int) -> NDArray[np.int64]:
    """Compute camera-pair shared observation counts.

    The coverage matrix is an (n_cameras, n_cameras) symmetric matrix where:
    - Diagonal [i, i]: Total observations from camera i
    - Off-diagonal [i, j]: Count of (sync_index, point_id) pairs seen by BOTH cameras i and j

    This reveals the pose network topology. Camera pairs with zero shared observations
    cannot be directly linked during stereo calibration.

    Args:
        image_points: ImagePoints to analyze
        n_cameras: Number of cameras (assumes ports 0 to n_cameras-1)

    Returns:
        (n_cameras, n_cameras) symmetric matrix of observation counts
    """
    df = image_points.df
    coverage = np.zeros((n_cameras, n_cameras), dtype=np.int64)

    # Group by (sync_index, point_id) to find which cameras see each point
    grouped = df.groupby(["sync_index", "point_id"])["port"].apply(set)

    for ports in grouped:
        port_list = sorted(ports)
        for i, port_i in enumerate(port_list):
            for port_j in port_list[i:]:
                if port_i < n_cameras and port_j < n_cameras:
                    coverage[port_i, port_j] += 1
                    if port_i != port_j:
                        coverage[port_j, port_i] += 1

    return coverage
