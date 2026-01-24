"""Tests for CalibrationObject."""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.synthetic import CalibrationObject


class TestConstruction:
    """Test CalibrationObject construction and validation."""

    def test_from_points_creates_valid_object(self):
        """Create object from arbitrary point cloud."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

        obj = CalibrationObject.from_points(points)

        assert obj.n_points == 4
        assert np.allclose(obj.points, points)
        assert np.array_equal(obj.point_ids, np.arange(4))

    def test_from_points_with_custom_ids(self):
        """Create object with custom point IDs."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        ids = np.array([10, 20, 30, 40], dtype=np.int64)

        obj = CalibrationObject.from_points(points, point_ids=ids)

        assert np.array_equal(obj.point_ids, ids)

    def test_invalid_points_rejected(self):
        """Reject invalid point arrays or IDs."""
        # Wrong shape
        bad_points = np.array([[1, 2], [3, 4]], dtype=np.float64)
        with pytest.raises(ValueError, match="Points must be \\(N, 3\\)"):
            CalibrationObject.from_points(bad_points)

        # Mismatched IDs
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        ids = np.array([0, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="same length"):
            CalibrationObject.from_points(points, point_ids=ids)


class TestPlanarGrid:
    """Test CalibrationObject.planar_grid() factory method."""

    def test_creates_correct_number_of_points(self):
        """Grid has rows * cols points."""
        obj = CalibrationObject.planar_grid(rows=3, cols=4, spacing_mm=10.0)

        assert obj.n_points == 12

    def test_points_are_coplanar_z_zero(self):
        """All grid points have Z=0."""
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=20.0)

        assert np.allclose(obj.points[:, 2], 0.0)

    def test_spacing_is_correct(self):
        """Adjacent points are separated by spacing_mm."""
        spacing = 25.0
        obj = CalibrationObject.planar_grid(rows=3, cols=3, spacing_mm=spacing)

        # Check horizontal spacing (points 0 and 1)
        dx = obj.points[1, 0] - obj.points[0, 0]
        assert np.isclose(dx, spacing)

        # Check vertical spacing (points 0 and 3)
        dy = obj.points[3, 1] - obj.points[0, 1]
        assert np.isclose(dy, spacing)

    def test_invalid_grid_rejected(self):
        """Grid must be at least 2x2 with positive spacing."""
        with pytest.raises(ValueError, match="at least 2x2"):
            CalibrationObject.planar_grid(rows=1, cols=5, spacing_mm=10.0)

        with pytest.raises(ValueError, match="must be positive"):
            CalibrationObject.planar_grid(rows=3, cols=3, spacing_mm=0.0)


class TestProperties:
    """Test CalibrationObject properties."""

    def test_n_points_property(self):
        """n_points returns number of points."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
        obj = CalibrationObject.from_points(points)

        assert obj.n_points == 5

    def test_centroid_is_center_of_mass(self):
        """Centroid is mean of all points."""
        points = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float64)
        obj = CalibrationObject.from_points(points)

        expected_centroid = np.array([2.5, 2.5, 2.5], dtype=np.float64)
        assert np.allclose(obj.centroid, expected_centroid)

    def test_extent_is_max_distance_from_centroid(self):
        """Extent is radius of bounding sphere."""
        # Unit cube corners
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        obj = CalibrationObject.from_points(points)

        # Centroid is at (0.5, 0.5, 0.5)
        # Farthest corners are at distance sqrt(3)/2 ≈ 0.866
        expected_extent = np.sqrt(3) / 2
        assert np.isclose(obj.extent, expected_extent)


if __name__ == "__main__":
    """Debug harness for running tests with saved outputs."""
    import sys

    # Run pytest on this file
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
