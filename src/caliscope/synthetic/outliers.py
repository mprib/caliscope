"""Outlier injection for synthetic image observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from caliscope.core.point_data import ImagePoints


@dataclass(frozen=True)
class OutlierConfig:
    fraction: float = 0.05
    magnitude_range: tuple[float, float] = (10.0, 50.0)
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.fraction <= 1:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")
        lo, hi = self.magnitude_range
        if lo >= hi:
            raise ValueError(f"magnitude_range must be (lo, hi) with lo < hi, got {self.magnitude_range}")


def inject_outliers(image_points: ImagePoints, config: OutlierConfig) -> tuple[ImagePoints, NDArray[np.int64]]:
    """Displace a random subset of observations by a gross pixel error.

    Returns the corrupted ImagePoints and the positional row indices
    that were corrupted.
    """
    df = image_points.df
    n = len(df)
    n_corrupt = round(config.fraction * n)

    if n_corrupt == 0:
        return ImagePoints(df), np.array([], dtype=np.int64)

    rng = np.random.default_rng(config.random_seed)
    indices = np.sort(rng.choice(n, size=n_corrupt, replace=False))

    lo, hi = config.magnitude_range
    magnitudes = rng.uniform(lo, hi, size=n_corrupt)
    angles = rng.uniform(0, 2 * np.pi, size=n_corrupt)

    x_vals: NDArray[np.float64] = df["img_loc_x"].to_numpy().copy()
    y_vals: NDArray[np.float64] = df["img_loc_y"].to_numpy().copy()
    x_vals[indices] += magnitudes * np.cos(angles)
    y_vals[indices] += magnitudes * np.sin(angles)
    df["img_loc_x"] = x_vals
    df["img_loc_y"] = y_vals

    return ImagePoints(df), indices.astype(np.int64)
