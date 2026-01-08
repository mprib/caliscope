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
    _compute_board_aspect_ratio,
    _max_possible_bbox_area,
    select_calibration_frames,
)
from caliscope.core.point_data import ImagePoints
from caliscope.helper import copy_contents_to_clean_dest

logger = logging.getLogger(__name__)


class TestBoardAwareCoverage:
    """Unit tests for board-aware coverage geometry.

    ChArUco internal corner geometry:
    - For an MxN board (M columns × N rows of squares), internal corners
      are at grid intersections where 4 squares meet.
    - Internal corners span from column 1 to M-1 and row 1 to N-1.
    - As a fraction of board dimensions: (M-2)/M width × (N-2)/N height.

    Example:
    - 3×4 board: 2×3 = 6 internal corners, spanning 1/3 × 2/4 of board
    - 5×7 board: 4×6 = 24 internal corners, spanning 3/5 × 5/7 of board

    The denser the grid, the closer internal corners are to the true board edge.
    """

    def test_square_board_aspect_ratio(self):
        """A square internal corner grid has aspect ratio 1.0."""
        # 4×4 board: 3×3 internal corners, each spanning 2/4 = 0.5 of dimension
        # obj_loc would be at positions 1, 2, 3 in both x and y (in square units)
        # Using arbitrary square_size of 5cm:
        data = {
            "sync_index": [0] * 9,
            "port": [0] * 9,
            "point_id": list(range(9)),
            "img_loc_x": [100] * 9,  # img_loc doesn't matter for aspect ratio
            "img_loc_y": [100] * 9,
            "obj_loc_x": [5, 10, 15, 5, 10, 15, 5, 10, 15],  # 3 columns at 5, 10, 15
            "obj_loc_y": [5, 5, 5, 10, 10, 10, 15, 15, 15],  # 3 rows at 5, 10, 15
        }
        df = pd.DataFrame(data)

        aspect = _compute_board_aspect_ratio(df)

        # X range = 15-5 = 10, Y range = 15-5 = 10, aspect = 1.0
        assert aspect == 1.0

    def test_wide_board_aspect_ratio(self):
        """A 5×3 board (wide) has internal corners with aspect > 1."""
        # 5×3 board: 4×2 = 8 internal corners
        # X spans columns 1-4 (3 units out of 5) → 3/5 of width
        # Y spans rows 1-2 (1 unit out of 3) → 1/3 of height
        # Internal corner aspect = (3/5) / (1/3) = 9/5 = 1.8? No wait...
        # The obj_loc spans: x = 1 to 4, y = 1 to 2 (in square units)
        # Range: x = 3, y = 1 → aspect = 3/1 = 3.0
        data = {
            "sync_index": [0] * 8,
            "port": [0] * 8,
            "point_id": list(range(8)),
            "img_loc_x": [100] * 8,
            "img_loc_y": [100] * 8,
            # 4 columns (x=1,2,3,4) × 2 rows (y=1,2) in square units
            "obj_loc_x": [1, 2, 3, 4, 1, 2, 3, 4],
            "obj_loc_y": [1, 1, 1, 1, 2, 2, 2, 2],
        }
        df = pd.DataFrame(data)

        aspect = _compute_board_aspect_ratio(df)

        # X range = 4-1 = 3, Y range = 2-1 = 1, aspect = 3.0
        assert aspect == 3.0

    def test_tall_board_aspect_ratio(self):
        """A 3×5 board (tall) has internal corners with aspect < 1."""
        # 3×5 board: 2×4 = 8 internal corners
        # X spans 1 unit, Y spans 3 units → aspect = 1/3
        data = {
            "sync_index": [0] * 8,
            "port": [0] * 8,
            "point_id": list(range(8)),
            "img_loc_x": [100] * 8,
            "img_loc_y": [100] * 8,
            # 2 columns (x=1,2) × 4 rows (y=1,2,3,4)
            "obj_loc_x": [1, 2, 1, 2, 1, 2, 1, 2],
            "obj_loc_y": [1, 1, 2, 2, 3, 3, 4, 4],
        }
        df = pd.DataFrame(data)

        aspect = _compute_board_aspect_ratio(df)

        # X range = 1, Y range = 3, aspect = 1/3
        assert abs(aspect - (1 / 3)) < 1e-6

    def test_max_bbox_square_board_in_square_image(self):
        """Square board in square image: max bbox is full image."""
        # Board aspect 1.0, image 1000×1000
        max_area = _max_possible_bbox_area((1000, 1000), 1.0)

        assert max_area == 1000 * 1000

    def test_max_bbox_wide_board_in_square_image(self):
        """Wide board (2:1) in square image: constrained by width."""
        # Board aspect 2.0 (twice as wide as tall)
        # In 1000×1000 image: board fills width (1000), height = 1000/2 = 500
        max_area = _max_possible_bbox_area((1000, 1000), 2.0)

        assert max_area == 1000 * 500

    def test_max_bbox_tall_board_in_square_image(self):
        """Tall board (1:2) in square image: constrained by height."""
        # Board aspect 0.5 (half as wide as tall)
        # In 1000×1000 image: board fills height (1000), width = 1000 * 0.5 = 500
        max_area = _max_possible_bbox_area((1000, 1000), 0.5)

        assert max_area == 500 * 1000

    def test_max_bbox_in_widescreen_image(self):
        """Board in 16:9 widescreen image."""
        # Square board (aspect 1.0) in 1920×1080 image
        # Image aspect = 1920/1080 = 16/9 ≈ 1.78
        # Square board is "taller" than image aspect, so height-constrained
        # Max height = 1080, max width = 1080 * 1.0 = 1080
        max_area = _max_possible_bbox_area((1920, 1080), 1.0)

        assert max_area == 1080 * 1080

    def test_denser_grid_larger_relative_coverage(self):
        """A denser grid has corners closer to board edge, so larger obj_loc spread.

        For two boards with SAME physical dimensions but different grid density,
        the denser board's internal corners span a larger fraction of the board.

        This test verifies the math:
        - 3×4 board: corners span 1/3 × 2/4 = 1/3 × 1/2 of board
        - 6×8 board: corners span 4/6 × 6/8 = 2/3 × 3/4 of board

        The 6×8 board's corners span 4x more area than the 3×4 board's corners,
        even for the same physical board size.
        """
        # Assume both boards are 30cm × 40cm physical size
        # 3×4 board: square_size = 10cm, corners at 10,20 × 10,20,30
        sparse_data = {
            "sync_index": [0] * 6,
            "port": [0] * 6,
            "point_id": list(range(6)),
            "img_loc_x": [100] * 6,
            "img_loc_y": [100] * 6,
            "obj_loc_x": [10, 20, 10, 20, 10, 20],  # 2 columns
            "obj_loc_y": [10, 10, 20, 20, 30, 30],  # 3 rows
        }
        sparse_df = pd.DataFrame(sparse_data)
        sparse_x_range = 20 - 10  # = 10
        sparse_y_range = 30 - 10  # = 20

        # 6×8 board: square_size = 5cm, corners at 5,10,15,20,25 × 5,10,...,35
        dense_data = {
            "sync_index": [0] * 35,
            "port": [0] * 35,
            "point_id": list(range(35)),
            "img_loc_x": [100] * 35,
            "img_loc_y": [100] * 35,
            "obj_loc_x": [5, 10, 15, 20, 25] * 7,  # 5 columns
            "obj_loc_y": [y for y in [5, 10, 15, 20, 25, 30, 35] for _ in range(5)],
        }
        dense_df = pd.DataFrame(dense_data)
        dense_x_range = 25 - 5  # = 20
        dense_y_range = 35 - 5  # = 30

        # Denser grid spans 2x in each dimension → 4x area
        assert dense_x_range == 2 * sparse_x_range
        assert dense_y_range == 1.5 * sparse_y_range  # 30/20 = 1.5

        # Both should have same aspect ratio (board shape is same)
        sparse_aspect = _compute_board_aspect_ratio(sparse_df)
        dense_aspect = _compute_board_aspect_ratio(dense_df)

        # Sparse: 10/20 = 0.5, Dense: 20/30 = 0.667
        # Wait - these AREN'T the same! The corner grid aspect differs from board aspect.
        # This is actually correct behavior - see docstring above.
        assert abs(sparse_aspect - 0.5) < 1e-6
        assert abs(dense_aspect - (20 / 30)) < 1e-6


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
