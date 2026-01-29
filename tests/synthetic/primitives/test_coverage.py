"""Tests for coverage matrix computation and structural analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliscope.core.point_data import ImagePoints
from caliscope.core.coverage_analysis import (
    analyze_multi_camera_coverage,
    classify_link_quality,
    compute_coverage_matrix,
    detect_structural_warnings,
    LinkQuality,
    WarningSeverity,
)
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


class TestLinkQualityClassification:
    """Test link quality classification based on observation count."""

    def test_good_quality(self):
        """>=200 observations is GOOD quality."""
        assert classify_link_quality(200) == LinkQuality.GOOD
        assert classify_link_quality(500) == LinkQuality.GOOD
        assert classify_link_quality(1000) == LinkQuality.GOOD

    def test_marginal_quality(self):
        """50-199 observations is MARGINAL quality."""
        assert classify_link_quality(50) == LinkQuality.MARGINAL
        assert classify_link_quality(100) == LinkQuality.MARGINAL
        assert classify_link_quality(199) == LinkQuality.MARGINAL

    def test_insufficient_quality(self):
        """<50 observations is INSUFFICIENT quality."""
        assert classify_link_quality(0) == LinkQuality.INSUFFICIENT
        assert classify_link_quality(10) == LinkQuality.INSUFFICIENT
        assert classify_link_quality(49) == LinkQuality.INSUFFICIENT


class TestExtrinsicCoverageReport:
    """Test the ExtrinsicCoverageReport structure."""

    def test_well_connected_network(self):
        """A fully connected 3-camera network has no structural issues."""
        image_points = make_simple_image_points()
        report = analyze_multi_camera_coverage(image_points)

        assert report.n_cameras == 3
        assert report.n_connected_components == 1
        assert report.isolated_cameras == []
        assert report.has_critical_issues is False

    def test_isolated_camera_detected(self):
        """An isolated camera is detected and reported."""
        # Create data where camera 0 has observations but no overlap with others
        rows = []
        for frame in range(3):
            # Camera 0 sees points 0-2 (unique to camera 0)
            for point_id in range(3):
                rows.append(
                    {
                        "sync_index": frame,
                        "port": 0,
                        "point_id": point_id,
                        "img_loc_x": 100.0,
                        "img_loc_y": 100.0,
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )
            # Cameras 1 and 2 share points 10-12
            for point_id in range(10, 13):
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

        df = pd.DataFrame(rows)
        image_points = ImagePoints(df)
        report = analyze_multi_camera_coverage(image_points)

        assert 0 in report.isolated_cameras
        assert report.has_critical_issues is True

    def test_multiple_components_detected(self):
        """Multiple disconnected components are detected."""
        # Create two isolated pairs of cameras (0,1) and (2,3)
        rows = []
        for frame in range(3):
            # Cameras 0 and 1 share points 0-2
            for point_id in range(3):
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
            # Cameras 2 and 3 share points 3-5 (different point IDs = no cross-link)
            for point_id in range(3, 6):
                for port in [2, 3]:
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

        report = analyze_multi_camera_coverage(image_points)

        assert report.n_connected_components == 2
        assert report.has_critical_issues is True

    def test_leaf_cameras_detected(self):
        """Cameras with only one connection are detected as leaf nodes."""
        # Create a chain: 0 -- 1 -- 2
        rows = []
        for frame in range(3):
            # Camera 0 and 1 share points 0-2
            for point_id in range(3):
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
            # Camera 1 and 2 share points 3-5
            for point_id in range(3, 6):
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

        df = pd.DataFrame(rows)
        image_points = ImagePoints(df)

        report = analyze_multi_camera_coverage(image_points)

        # Cameras 0 and 2 are leaf nodes (only connected to camera 1)
        leaf_ports = [port for port, _, _ in report.leaf_cameras]
        assert 0 in leaf_ports
        assert 2 in leaf_ports
        assert 1 not in leaf_ports  # Camera 1 has 2 connections


class TestStructuralWarnings:
    """Test structural warning detection."""

    def test_no_warnings_for_good_network(self):
        """A well-connected network produces no warnings."""
        image_points = make_simple_image_points()
        report = analyze_multi_camera_coverage(image_points)

        warnings = detect_structural_warnings(report, n_cameras=3)

        # The network in make_simple_image_points is fully connected (no leaves)
        # so there should be no critical or warning messages
        critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
        assert len(critical) == 0

    def test_isolated_camera_critical_warning(self):
        """Isolated camera produces a CRITICAL warning."""
        # Create data where camera 0 has observations but no overlap with others
        rows = []
        for frame in range(3):
            # Camera 0 sees points 0-2 (unique to camera 0)
            for point_id in range(3):
                rows.append(
                    {
                        "sync_index": frame,
                        "port": 0,
                        "point_id": point_id,
                        "img_loc_x": 100.0,
                        "img_loc_y": 100.0,
                        "obj_loc_x": 0.0,
                        "obj_loc_y": 0.0,
                        "obj_loc_z": 0.0,
                    }
                )
            # Cameras 1 and 2 share points 10-12
            for point_id in range(10, 13):
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

        df = pd.DataFrame(rows)
        image_points = ImagePoints(df)
        report = analyze_multi_camera_coverage(image_points)

        warnings = detect_structural_warnings(report, n_cameras=3)

        critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
        assert len(critical) >= 1
        assert any("C0" in w.message for w in critical)

    def test_two_camera_setup_no_leaf_warnings(self):
        """A 2-camera setup doesn't warn about leaf nodes (both are necessarily leaves)."""
        rows = []
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
        report = analyze_multi_camera_coverage(image_points)

        # With n_cameras=2, leaf warnings are suppressed
        warnings = detect_structural_warnings(report, n_cameras=2)

        # No warnings at all for a good 2-camera setup
        assert len(warnings) == 0

    def test_weak_leaf_produces_warning(self):
        """A leaf camera with few observations produces a WARNING."""
        # Create a chain with weak link: 0 --weak-- 1 -- 2
        rows = []
        for frame in range(3):
            # Camera 0 and 1 share only 3 points (9 total observations - weak)
            for point_id in range(3):
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
            # Camera 1 and 2 share points 3-52 (150 observations - strong)
            for point_id in range(3, 53):
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

        df = pd.DataFrame(rows)
        image_points = ImagePoints(df)
        report = analyze_multi_camera_coverage(image_points)

        warnings = detect_structural_warnings(report, n_cameras=3, min_leaf_observations=100)

        # Camera 0 is a weak leaf (9 observations < 100 threshold)
        warning_messages = [w.message for w in warnings if w.severity == WarningSeverity.WARNING]
        assert any("C0" in msg for msg in warning_messages)

    def test_warnings_sorted_by_severity(self):
        """Warnings are returned sorted by severity (CRITICAL first)."""
        # Create multiple issues: isolated camera + multiple components
        rows = []
        # Only camera 1 and 2 connected, camera 0 isolated
        for frame in range(3):
            for point_id in range(3):
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
            # Camera 0 sees different points (no shared with 1 or 2)
            for point_id in range(10, 13):
                rows.append(
                    {
                        "sync_index": frame,
                        "port": 0,
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
        report = analyze_multi_camera_coverage(image_points)

        warnings = detect_structural_warnings(report, n_cameras=3)

        # All CRITICAL warnings should come before WARNING and INFO
        severities = [w.severity for w in warnings]
        critical_idx = [i for i, s in enumerate(severities) if s == WarningSeverity.CRITICAL]
        warning_idx = [i for i, s in enumerate(severities) if s == WarningSeverity.WARNING]
        info_idx = [i for i, s in enumerate(severities) if s == WarningSeverity.INFO]

        if critical_idx and warning_idx:
            assert max(critical_idx) < min(warning_idx)
        if critical_idx and info_idx:
            assert max(critical_idx) < min(info_idx)
        if warning_idx and info_idx:
            assert max(warning_idx) < min(info_idx)


if __name__ == "__main__":
    """Debug harness for running tests with saved outputs."""
    import sys

    # Run pytest on this file
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
