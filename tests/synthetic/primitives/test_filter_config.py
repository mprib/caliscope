"""Tests for FilterConfig observation filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from caliscope.core.point_data import ImagePoints
from caliscope.synthetic import FilterConfig


def make_test_image_points() -> ImagePoints:
    """Create test ImagePoints with predictable structure for filter testing.

    4 cameras (ports 0-3), 5 frames (0-4):
    - Points 0-2: camera 0 only (exclusive to cam 0)
    - Points 3-5: camera 1 only (exclusive to cam 1)
    - Points 6-8: cameras 0 and 1 shared (linkage 0-1)
    - Points 9-11: cameras 2 and 3 shared (linkage 2-3)
    - Points 12-14: all cameras (cross-linkage)

    Total per frame: 3+3+6+6+12 = 30 observations
    Total: 30 * 5 frames = 150 observations

    This allows testing killed_linkages while preserving non-shared data.
    """
    rows = []
    for frame in range(5):
        # Points 0-2: camera 0 only
        for point_id in range(3):
            rows.append(
                {
                    "sync_index": frame,
                    "port": 0,
                    "point_id": point_id,
                    "img_loc_x": float(frame * 10 + point_id),
                    "img_loc_y": float(frame * 10 + point_id),
                    "obj_loc_x": 0.0,
                    "obj_loc_y": 0.0,
                    "obj_loc_z": 0.0,
                }
            )
        # Points 3-5: camera 1 only
        for point_id in range(3, 6):
            rows.append(
                {
                    "sync_index": frame,
                    "port": 1,
                    "point_id": point_id,
                    "img_loc_x": float(frame * 10 + point_id),
                    "img_loc_y": float(frame * 10 + point_id),
                    "obj_loc_x": 0.0,
                    "obj_loc_y": 0.0,
                    "obj_loc_z": 0.0,
                }
            )
        # Points 6-8: cameras 0 and 1 shared
        for point_id in range(6, 9):
            for port in [0, 1]:
                rows.append(
                    {
                        "sync_index": frame,
                        "port": port,
                        "point_id": point_id,
                        "img_loc_x": float(frame * 10 + port + point_id),
                        "img_loc_y": float(frame * 10 + port + point_id),
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )
        # Points 9-11: cameras 2 and 3 shared
        for point_id in range(9, 12):
            for port in [2, 3]:
                rows.append(
                    {
                        "sync_index": frame,
                        "port": port,
                        "point_id": point_id,
                        "img_loc_x": float(frame * 10 + port + point_id),
                        "img_loc_y": float(frame * 10 + port + point_id),
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )
        # Points 12-14: all cameras
        for point_id in range(12, 15):
            for port in range(4):
                rows.append(
                    {
                        "sync_index": frame,
                        "port": port,
                        "point_id": point_id,
                        "img_loc_x": float(frame * 10 + port + point_id),
                        "img_loc_y": float(frame * 10 + port + point_id),
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )

    df = pd.DataFrame(rows)
    return ImagePoints(df)


class TestConstruction:
    """Test FilterConfig construction and validation."""

    def test_empty_config_is_valid(self):
        """Empty config (no filters) is valid."""
        config = FilterConfig()

        assert config.dropped_cameras == ()
        assert config.killed_linkages == ()
        assert config.dropped_frame_ranges == ()
        assert config.random_dropout_fraction == 0.0
        assert config.random_seed == 42

    def test_invalid_dropout_fraction_rejected(self):
        """Dropout fraction must be in [0, 1)."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\)"):
            FilterConfig(random_dropout_fraction=-0.1)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\)"):
            FilterConfig(random_dropout_fraction=1.0)

    def test_invalid_killed_linkage_rejected(self):
        """Cannot kill linkage between a camera and itself."""
        with pytest.raises(ValueError, match="Cannot kill linkage with same camera"):
            FilterConfig(killed_linkages=((0, 0),))


class TestApplyEmpty:
    """Test that empty config returns unmodified data."""

    def test_empty_config_no_filtering(self):
        """Empty config returns identical data."""
        image_points = make_test_image_points()
        config = FilterConfig()

        filtered = config.apply(image_points)

        # Should have same number of rows
        assert len(filtered.df) == len(image_points.df)

        # Data should be identical
        pd.testing.assert_frame_equal(filtered.df, image_points.df)


class TestDroppedCameras:
    """Test dropped_cameras filter."""

    def test_drop_single_camera(self):
        """Drop all observations from one camera."""
        image_points = make_test_image_points()
        original_count = len(image_points.df)
        config = FilterConfig(dropped_cameras=(1,))

        filtered = config.apply(image_points)

        # Should have no observations from port 1
        assert 1 not in filtered.df["port"].values

        # Other cameras should still be present
        assert 0 in filtered.df["port"].values
        assert 2 in filtered.df["port"].values
        assert 3 in filtered.df["port"].values

        # Verify some data was removed
        assert len(filtered.df) < original_count


class TestKilledLinkages:
    """Test killed_linkages filter."""

    def test_kill_linkage_removes_shared_observations(self):
        """Killing linkage between cameras removes shared points."""
        image_points = make_test_image_points()
        config = FilterConfig(killed_linkages=((0, 1),))

        filtered = config.apply(image_points)

        # Should still have data from both cameras
        assert 0 in filtered.df["port"].values
        assert 1 in filtered.df["port"].values

        # But no (sync_index, point_id) should be shared between them
        cam0_obs = set(
            zip(
                filtered.df[filtered.df["port"] == 0]["sync_index"],
                filtered.df[filtered.df["port"] == 0]["point_id"],
            )
        )
        cam1_obs = set(
            zip(
                filtered.df[filtered.df["port"] == 1]["sync_index"],
                filtered.df[filtered.df["port"] == 1]["point_id"],
            )
        )

        shared = cam0_obs & cam1_obs
        assert len(shared) == 0, "Killed linkage should have no shared observations"


class TestDroppedFrameRanges:
    """Test dropped_frame_ranges filter."""

    def test_drop_single_frame_range(self):
        """Drop observations in a frame range."""
        image_points = make_test_image_points()
        config = FilterConfig(dropped_frame_ranges=((1, 3),))  # Drop frames 1, 2, 3

        filtered = config.apply(image_points)

        # Frames 1, 2, 3 should be gone
        assert 1 not in filtered.df["sync_index"].values
        assert 2 not in filtered.df["sync_index"].values
        assert 3 not in filtered.df["sync_index"].values

        # Frames 0 and 4 should remain
        assert 0 in filtered.df["sync_index"].values
        assert 4 in filtered.df["sync_index"].values


class TestRandomDropout:
    """Test random_dropout_fraction filter."""

    def test_zero_dropout_keeps_all_data(self):
        """Zero dropout fraction keeps all data."""
        image_points = make_test_image_points()
        config = FilterConfig(random_dropout_fraction=0.0)

        filtered = config.apply(image_points)

        assert len(filtered.df) == len(image_points.df)

    def test_dropout_removes_approximately_correct_fraction(self):
        """Dropout removes approximately the specified fraction."""
        image_points = make_test_image_points()
        config = FilterConfig(random_dropout_fraction=0.3, random_seed=42)

        filtered = config.apply(image_points)

        original_count = len(image_points.df)
        filtered_count = len(filtered.df)

        # Should drop ~30% (with some statistical variance)
        dropped_fraction = 1 - (filtered_count / original_count)
        assert 0.2 < dropped_fraction < 0.4, f"Expected ~30% dropout, got {dropped_fraction:.1%}"

    def test_dropout_is_reproducible(self):
        """Same seed produces same dropout pattern."""
        image_points = make_test_image_points()
        config = FilterConfig(random_dropout_fraction=0.5, random_seed=123)

        filtered1 = config.apply(image_points)
        filtered2 = config.apply(image_points)

        # Same rows should be kept
        pd.testing.assert_frame_equal(filtered1.df, filtered2.df)


class TestCombinedFilters:
    """Test that multiple filters can be applied together."""

    def test_all_filters_combined(self):
        """All filters can be applied together."""
        image_points = make_test_image_points()
        config = FilterConfig(
            dropped_cameras=(3,),
            killed_linkages=((0, 1),),
            dropped_frame_ranges=((4, 4),),
            random_dropout_fraction=0.1,
            random_seed=42,
        )

        filtered = config.apply(image_points)

        # Should have applied all filters
        assert 3 not in filtered.df["port"].values
        assert 4 not in filtered.df["sync_index"].values

        # Some data should remain
        assert len(filtered.df) > 0


class TestImmutability:
    """Test that FilterConfig is immutable and returns new instances."""

    def test_with_killed_linkage_returns_new_instance(self):
        """with_killed_linkage returns a new FilterConfig."""
        config1 = FilterConfig()
        config2 = config1.with_killed_linkage(0, 1)

        assert config1 is not config2
        assert config1.killed_linkages == ()
        assert config2.killed_linkages == ((0, 1),)

    def test_with_killed_linkage_normalizes_order(self):
        """with_killed_linkage normalizes (a, b) to (min, max)."""
        config = FilterConfig()

        config_ab = config.with_killed_linkage(0, 1)
        config_ba = config.with_killed_linkage(1, 0)

        # Both should produce same linkage tuple
        assert config_ab.killed_linkages == ((0, 1),)
        assert config_ba.killed_linkages == ((0, 1),)

    def test_without_killed_linkage_returns_new_instance(self):
        """without_killed_linkage returns a new FilterConfig."""
        config1 = FilterConfig(killed_linkages=((0, 1), (2, 3)))
        config2 = config1.without_killed_linkage(0, 1)

        assert config1 is not config2
        assert len(config1.killed_linkages) == 2
        assert config2.killed_linkages == ((2, 3),)


if __name__ == "__main__":
    """Debug harness for running tests with saved outputs."""
    import sys

    # Run pytest on this file
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
