"""Tests for outlier injection."""

import numpy as np
import pytest

from caliscope.synthetic.outliers import OutlierConfig, inject_outliers
from caliscope.synthetic.scene_factories import default_ring_scene


@pytest.fixture
def image_points():
    return default_ring_scene().image_points_noisy


class TestOutlierConfig:
    def test_rejects_negative_fraction(self) -> None:
        with pytest.raises(ValueError, match="fraction must be in"):
            OutlierConfig(fraction=-0.1)

    def test_rejects_fraction_above_one(self) -> None:
        with pytest.raises(ValueError, match="fraction must be in"):
            OutlierConfig(fraction=1.5)

    def test_rejects_inverted_magnitude_range(self) -> None:
        with pytest.raises(ValueError, match="magnitude_range must be"):
            OutlierConfig(magnitude_range=(50.0, 10.0))

    def test_rejects_equal_magnitude_range(self) -> None:
        with pytest.raises(ValueError, match="magnitude_range must be"):
            OutlierConfig(magnitude_range=(10.0, 10.0))


class TestInjectOutliers:
    def test_corrupted_count_matches_fraction(self, image_points) -> None:
        for frac in [0.05, 0.1, 0.5]:
            config = OutlierConfig(fraction=frac)
            _, indices = inject_outliers(image_points, config)
            expected = round(frac * len(image_points.df))
            assert len(indices) == expected

    def test_magnitudes_within_range(self, image_points) -> None:
        config = OutlierConfig(fraction=0.1, magnitude_range=(15.0, 40.0))
        corrupted, indices = inject_outliers(image_points, config)

        orig_df = image_points.df
        corr_df = corrupted.df

        dx = corr_df.iloc[indices]["img_loc_x"].values - orig_df.iloc[indices]["img_loc_x"].values
        dy = corr_df.iloc[indices]["img_loc_y"].values - orig_df.iloc[indices]["img_loc_y"].values
        mags = np.sqrt(dx**2 + dy**2)

        assert np.all(mags >= 15.0 - 1e-10)
        assert np.all(mags <= 40.0 + 1e-10)

    def test_reproducible_with_same_seed(self, image_points) -> None:
        config = OutlierConfig(fraction=0.05, random_seed=99)
        c1, i1 = inject_outliers(image_points, config)
        c2, i2 = inject_outliers(image_points, config)

        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_array_equal(c1.df.values, c2.df.values)

    def test_different_seeds_differ(self, image_points) -> None:
        c1 = OutlierConfig(fraction=0.05, random_seed=1)
        c2 = OutlierConfig(fraction=0.05, random_seed=2)
        _, i1 = inject_outliers(image_points, c1)
        _, i2 = inject_outliers(image_points, c2)

        assert not np.array_equal(i1, i2)

    def test_uncorrupted_rows_identical(self, image_points) -> None:
        config = OutlierConfig(fraction=0.1)
        corrupted, indices = inject_outliers(image_points, config)

        orig_df = image_points.df
        corr_df = corrupted.df

        mask = np.ones(len(orig_df), dtype=bool)
        mask[indices] = False

        np.testing.assert_array_equal(
            orig_df.iloc[mask].values,
            corr_df.iloc[mask].values,
        )

    def test_obj_loc_untouched(self, image_points) -> None:
        config = OutlierConfig(fraction=0.2)
        corrupted, _ = inject_outliers(image_points, config)

        for col in ["obj_loc_x", "obj_loc_y", "obj_loc_z"]:
            np.testing.assert_array_equal(
                image_points.df[col].values,
                corrupted.df[col].values,
            )

    def test_zero_fraction_returns_identical(self, image_points) -> None:
        config = OutlierConfig(fraction=0.0)
        corrupted, indices = inject_outliers(image_points, config)

        assert len(indices) == 0
        np.testing.assert_array_equal(image_points.df.values, corrupted.df.values)

    def test_full_fraction_corrupts_all(self, image_points) -> None:
        config = OutlierConfig(fraction=1.0)
        _, indices = inject_outliers(image_points, config)

        assert len(indices) == len(image_points.df)


if __name__ == "__main__":
    scene = default_ring_scene()
    ip = scene.image_points_noisy
    print(f"Total observations: {len(ip.df)}")

    config = OutlierConfig(fraction=0.05, magnitude_range=(10.0, 50.0))
    corrupted, indices = inject_outliers(ip, config)

    orig_df = ip.df
    corr_df = corrupted.df

    dx = corr_df.iloc[indices]["img_loc_x"].values - orig_df.iloc[indices]["img_loc_x"].values
    dy = corr_df.iloc[indices]["img_loc_y"].values - orig_df.iloc[indices]["img_loc_y"].values
    mags = np.sqrt(dx**2 + dy**2)

    print(f"Corrupted: {len(indices)} rows ({100 * len(indices) / len(orig_df):.1f}%)")
    print(f"Magnitude range: [{mags.min():.1f}, {mags.max():.1f}] px")

    pytest.main([__file__, "-v"])
