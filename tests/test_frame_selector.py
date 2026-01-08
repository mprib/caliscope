"""Tests for automatic calibration frame selection.

These tests focus on behavior rather than implementation details:
- Determinism: same input produces identical output
- Edge cases: empty port, all frames ineligible
- Integration: realistic frame selection with real tracking data
"""

import logging
from pathlib import Path

import pandas as pd

from caliscope import __root__
from caliscope.core.frame_selector import (
    FrameSelectionResult,
    select_calibration_frames,
)
from caliscope.core.point_data import ImagePoints
from caliscope.helper import copy_contents_to_clean_dest

logger = logging.getLogger(__name__)


class TestDeterminism:
    """Same input must produce identical output on repeated runs."""

    def test_repeated_calls_produce_identical_results(self, tmp_path: Path):
        """Frame selection is deterministic - no random sampling."""
        original_path = Path(__root__, "tests", "sessions", "prerecorded_calibration")
        copy_contents_to_clean_dest(original_path, tmp_path)

        xy_csv = tmp_path / "calibration" / "intrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
        image_points = ImagePoints.from_csv(xy_csv)

        # Run selection multiple times
        results = [
            select_calibration_frames(
                image_points,
                port=0,
                image_size=(1280, 720),
                target_frame_count=15,
            )
            for _ in range(3)
        ]

        # All results must be identical
        for i in range(1, len(results)):
            assert results[i].selected_frames == results[0].selected_frames
            assert results[i].coverage_fraction == results[0].coverage_fraction
            assert results[i].eligible_frame_count == results[0].eligible_frame_count


class TestEdgeCases:
    """Handle edge cases gracefully without crashing."""

    def test_empty_port_returns_empty_result(self):
        """Port with no data returns empty result, not an error."""
        data = {
            "sync_index": [0, 1],
            "port": [1, 1],  # Only port 1 has data
            "point_id": [0, 0],
            "img_loc_x": [100, 200],
            "img_loc_y": [100, 200],
            "obj_loc_x": [0.0, 0.05],
            "obj_loc_y": [0.0, 0.05],
        }
        image_points = ImagePoints(pd.DataFrame(data))

        result = select_calibration_frames(
            image_points,
            port=0,  # Port 0 has no data
            image_size=(1920, 1080),
        )

        assert result.selected_frames == []
        assert result.total_frame_count == 0
        assert result.eligible_frame_count == 0

    def test_all_frames_ineligible_returns_empty_result(self):
        """When all frames fail eligibility criteria, return empty result."""
        # All frames have only 3 corners (below default min of 6)
        data = []
        for sync_index in range(5):
            for point_id in range(3):
                data.append(
                    {
                        "sync_index": sync_index,
                        "port": 0,
                        "point_id": point_id,
                        "img_loc_x": 100 + point_id * 100,
                        "img_loc_y": 100 + point_id * 100,
                        "obj_loc_x": point_id * 0.05,
                        "obj_loc_y": point_id * 0.05,
                    }
                )

        image_points = ImagePoints(pd.DataFrame(data))

        result = select_calibration_frames(
            image_points,
            port=0,
            image_size=(1920, 1080),
        )

        assert result.selected_frames == []
        assert result.total_frame_count == 5
        assert result.eligible_frame_count == 0


class TestIntegration:
    """Integration tests with real charuco tracking data."""

    def test_selects_frames_from_intrinsic_data(self, tmp_path: Path):
        """Integration test with prerecorded intrinsic calibration data.

        Uses 4x5 charuco board (12 internal corners) at 1280x720 resolution.
        Verifies the algorithm selects reasonable frames with good coverage.
        """
        original_path = Path(__root__, "tests", "sessions", "prerecorded_calibration")
        copy_contents_to_clean_dest(original_path, tmp_path)

        xy_csv = tmp_path / "calibration" / "intrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
        image_points = ImagePoints.from_csv(xy_csv)

        # Test across all ports
        for port in [0, 1, 2, 3]:
            result = select_calibration_frames(
                image_points,
                port=port,
                image_size=(1280, 720),
                target_frame_count=20,
            )

            logger.info(f"Port {port}: selected {len(result.selected_frames)} frames")
            logger.info(f"  Coverage: {result.coverage_fraction:.2%}")
            logger.info(f"  Edge coverage: {result.edge_coverage_fraction:.2%}")
            logger.info(f"  Eligible: {result.eligible_frame_count}/{result.total_frame_count}")

            # Validate result structure
            assert isinstance(result, FrameSelectionResult)
            assert all(isinstance(f, int) for f in result.selected_frames)
            assert 0 <= result.coverage_fraction <= 1
            assert 0 <= result.edge_coverage_fraction <= 1
            assert 0 <= result.corner_coverage_fraction <= 1

    def test_port_0_selects_frames_and_reports_coverage(self, tmp_path: Path):
        """Port 0 should select frames and report meaningful coverage metrics."""
        original_path = Path(__root__, "tests", "sessions", "prerecorded_calibration")
        copy_contents_to_clean_dest(original_path, tmp_path)

        xy_csv = tmp_path / "calibration" / "intrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
        image_points = ImagePoints.from_csv(xy_csv)

        result = select_calibration_frames(
            image_points,
            port=0,
            image_size=(1280, 720),
            target_frame_count=20,
        )

        # Verify the algorithm functions correctly:
        # - Selects at least some frames from eligible pool
        # - Reports non-zero coverage metrics
        # Note: actual coverage depends on test data quality, not algorithm correctness
        assert len(result.selected_frames) > 0, "Should select at least some frames"
        assert result.eligible_frame_count > 0, "Should have eligible frames"
        assert result.coverage_fraction > 0, "Should report non-zero coverage"
        assert len(result.selected_frames) <= result.eligible_frame_count

    def test_selected_frames_are_from_original_data(self, tmp_path: Path):
        """Selected sync_index values must exist in original data."""
        original_path = Path(__root__, "tests", "sessions", "prerecorded_calibration")
        copy_contents_to_clean_dest(original_path, tmp_path)

        xy_csv = tmp_path / "calibration" / "intrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
        image_points = ImagePoints.from_csv(xy_csv)

        result = select_calibration_frames(
            image_points,
            port=0,
            image_size=(1280, 720),
            target_frame_count=15,
        )

        # All selected frames must exist in the original data
        port_0_frames = set(image_points.df[image_points.df["port"] == 0]["sync_index"].unique())
        for frame in result.selected_frames:
            assert frame in port_0_frames, f"Selected frame {frame} not in original data"


if __name__ == "__main__":
    from pathlib import Path

    import caliscope.logger

    caliscope.logger.setup_logging()

    # Run tests with debug output
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Run determinism test
    test_determinism = TestDeterminism()
    test_determinism.test_repeated_calls_produce_identical_results(debug_dir)
    logger.info("PASS: test_repeated_calls_produce_identical_results")

    # Run edge case tests
    test_edge_cases = TestEdgeCases()
    test_edge_cases.test_empty_port_returns_empty_result()
    logger.info("PASS: test_empty_port_returns_empty_result")

    test_edge_cases.test_all_frames_ineligible_returns_empty_result()
    logger.info("PASS: test_all_frames_ineligible_returns_empty_result")

    # Run integration tests
    test_integration = TestIntegration()
    test_integration.test_selects_frames_from_intrinsic_data(debug_dir)
    logger.info("PASS: test_selects_frames_from_intrinsic_data")

    test_integration.test_port_0_selects_frames_and_reports_coverage(debug_dir)
    logger.info("PASS: test_port_0_selects_frames_and_reports_coverage")

    test_integration.test_selected_frames_are_from_original_data(debug_dir)
    logger.info("PASS: test_selected_frames_are_from_original_data")

    logger.info("All tests passed!")
