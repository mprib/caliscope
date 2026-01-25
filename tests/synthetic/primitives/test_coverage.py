"""Tests for coverage matrix computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliscope.core.point_data import ImagePoints
from caliscope.core.coverage_analysis import compute_coverage_matrix
from caliscope.synthetic import FilterConfig


# Mapping for 3-camera test setup (ports 0, 1, 2 -> indices 0, 1, 2)
THREE_CAMERA_PORT_TO_INDEX = {0: 0, 1: 1, 2: 2}


def make_simple_image_points() -> ImagePoints:
    """Create ImagePoints for coverage testing with partial overlap.

    3 cameras, 3 frames, structured visibility:
    - Points 0-1: seen by cameras 0 and 1 only (shared between 0,1)
    - Points 2-3: seen by cameras 1 and 2 only (shared between 1,2)
    - Points 4-5: seen by cameras 0 and 2 only (shared between 0,2)
    - Points 6-7: seen by all cameras (cross-linkage points)

    This allows testing killed linkages without removing all cross-camera links.
    """
    rows = []
    for frame in range(3):
        # Points 0-1: cameras 0,1 only
        for point_id in [0, 1]:
            for port in [0, 1]:
                rows.append(
                    {
                        "sync_index": frame,
                        "port": port,
                        "point_id": point_id,
                        "img_loc_x": 100.0,
                        "img_loc_y": 100.0,
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )
        # Points 2-3: cameras 1,2 only
        for point_id in [2, 3]:
            for port in [1, 2]:
                rows.append(
                    {
                        "sync_index": frame,
                        "port": port,
                        "point_id": point_id,
                        "img_loc_x": 100.0,
                        "img_loc_y": 100.0,
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )
        # Points 4-5: cameras 0,2 only
        for point_id in [4, 5]:
            for port in [0, 2]:
                rows.append(
                    {
                        "sync_index": frame,
                        "port": port,
                        "point_id": point_id,
                        "img_loc_x": 100.0,
                        "img_loc_y": 100.0,
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )
        # Points 6-7: all cameras
        for point_id in [6, 7]:
            for port in range(3):
                rows.append(
                    {
                        "sync_index": frame,
                        "port": port,
                        "point_id": point_id,
                        "img_loc_x": 100.0,
                        "img_loc_y": 100.0,
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )

    df = pd.DataFrame(rows)
    return ImagePoints(df)


class TestCoverageMatrix:
    """Test basic coverage matrix computation."""

    def test_coverage_matrix_shape(self):
        """Coverage matrix has shape (n_cameras, n_cameras)."""
        image_points = make_simple_image_points()
        coverage = compute_coverage_matrix(image_points, THREE_CAMERA_PORT_TO_INDEX)

        assert coverage.shape == (3, 3)

    def test_coverage_matrix_is_symmetric(self):
        """Coverage matrix is symmetric."""
        image_points = make_simple_image_points()
        coverage = compute_coverage_matrix(image_points, THREE_CAMERA_PORT_TO_INDEX)

        assert np.allclose(coverage, coverage.T)

    def test_diagonal_is_total_observations(self):
        """Diagonal elements are total observations per camera."""
        image_points = make_simple_image_points()
        coverage = compute_coverage_matrix(image_points, THREE_CAMERA_PORT_TO_INDEX)

        # Each camera sees 6 points * 3 frames = 18 observations
        expected_diagonal = 18

        assert coverage[0, 0] == expected_diagonal
        assert coverage[1, 1] == expected_diagonal
        assert coverage[2, 2] == expected_diagonal


class TestKilledLinkages:
    """Test coverage matrix with killed linkages."""

    def test_killed_linkage_shows_zero_shared(self):
        """Killed linkage results in zero shared observations between pair."""
        image_points = make_simple_image_points()

        # Kill linkage between cameras 0 and 1
        config = FilterConfig(killed_linkages=((0, 1),))
        filtered = config.apply(image_points)

        coverage = compute_coverage_matrix(filtered, THREE_CAMERA_PORT_TO_INDEX)

        # Cameras 0 and 1 should have zero shared observations
        assert coverage[0, 1] == 0
        assert coverage[1, 0] == 0

        # Other pairs should still have shared observations
        assert coverage[0, 2] > 0
        assert coverage[1, 2] > 0


class TestDroppedCameras:
    """Test coverage matrix with dropped cameras."""

    def test_dropped_camera_has_zero_observations(self):
        """Dropped camera shows zero in coverage matrix."""
        image_points = make_simple_image_points()

        config = FilterConfig(dropped_cameras=(1,))
        filtered = config.apply(image_points)

        # Pass full port_to_index to maintain 3x3 shape even with dropped camera
        coverage = compute_coverage_matrix(filtered, THREE_CAMERA_PORT_TO_INDEX)

        # Camera 1 should have zero observations (entire row/column is zero)
        assert coverage[1, 1] == 0
        assert coverage[0, 1] == 0
        assert coverage[1, 0] == 0
        assert coverage[1, 2] == 0
        assert coverage[2, 1] == 0

        # Cameras 0 and 2 should still have observations
        assert coverage[0, 0] > 0
        assert coverage[2, 2] > 0
        assert coverage[0, 2] > 0


class TestCoverageIntegration:
    """Test coverage matrix with realistic scenarios."""

    def test_stereo_pair_scenario(self):
        """Two cameras with good overlap (typical stereo calibration)."""
        rows = []

        # 10 frames, both cameras see 5 points each frame
        for frame in range(10):
            for point_id in range(5):
                for port in [0, 1]:
                    rows.append(
                        {
                            "sync_index": frame,
                            "port": port,
                            "point_id": point_id,
                            "img_loc_x": 100.0,
                            "img_loc_y": 100.0,
                            "obj_loc_x": 0.0,
                            "obj_loc_y": 0.0,
                            "obj_loc_z": 0.0,
                        }
                    )

        df = pd.DataFrame(rows)
        image_points = ImagePoints(df)

        two_camera_port_to_index = {0: 0, 1: 1}
        coverage = compute_coverage_matrix(image_points, two_camera_port_to_index)

        # Each camera: 10 frames * 5 points = 50 observations
        assert coverage[0, 0] == 50
        assert coverage[1, 1] == 50

        # Shared: all 50 observations
        assert coverage[0, 1] == 50
        assert coverage[1, 0] == 50


if __name__ == "__main__":
    """Debug harness for running tests with saved outputs."""
    import sys

    # Run pytest on this file
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
