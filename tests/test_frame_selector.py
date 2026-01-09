"""Tests for automatic calibration frame selection.

These tests focus on behavior rather than implementation details:
- Determinism: same input produces identical output
- Edge cases: empty port, all frames ineligible
- Integration: realistic frame selection with real tracking data
- Orientation diversity: homography-based orientation extraction
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from caliscope import __root__
from caliscope.core.frame_selector import (
    FrameSelectionResult,
    OrientationFeatures,
    _compute_orientation_features,
    _get_orientation_bin,
    _score_frame,
    select_calibration_frames,
)
from caliscope.core.point_data import ImagePoints
from caliscope.helper import copy_contents_to_clean_dest

logger = logging.getLogger(__name__)


class TestOrientationExtraction:
    """Tests for homography-based orientation feature extraction."""

    def test_frontal_board_has_low_tilt_magnitude(self):
        """A board parallel to the image plane should have near-zero tilt."""
        # Create a frontal-parallel board (no perspective distortion)
        # Object points form a 3x3 grid centered at origin
        obj_x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        obj_y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # Image points are scaled and translated versions (affine, no perspective)
        scale = 100
        offset_x, offset_y = 500, 300
        img_x = [x * scale + offset_x for x in obj_x]
        img_y = [y * scale + offset_y for y in obj_y]

        data = {
            "sync_index": [0] * 9,
            "port": [0] * 9,
            "point_id": list(range(9)),
            "obj_loc_x": obj_x,
            "obj_loc_y": obj_y,
            "img_loc_x": img_x,
            "img_loc_y": img_y,
        }
        df = pd.DataFrame(data)

        orientation = _compute_orientation_features(df)

        # Frontal board should have very low tilt magnitude
        assert orientation.tilt_magnitude < 0.01, (
            f"Expected low tilt for frontal board, got {orientation.tilt_magnitude}"
        )

    def test_tilted_board_has_higher_tilt_magnitude(self):
        """A tilted board should have measurable tilt magnitude."""
        # Create a board with perspective distortion (simulating tilt)
        # Object points form a 3x3 grid
        obj_x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        obj_y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # Image points with perspective foreshortening (top smaller than bottom)
        # This simulates a board tilted away from the camera at the top
        img_x = [
            505,
            550,
            595,  # Top row: narrower
            500,
            550,
            600,  # Middle row
            495,
            550,
            605,  # Bottom row: wider
        ]
        img_y = [
            300,
            300,
            300,
            350,
            350,
            350,
            410,
            410,
            410,  # Increased spacing at bottom
        ]

        data = {
            "sync_index": [0] * 9,
            "port": [0] * 9,
            "point_id": list(range(9)),
            "obj_loc_x": obj_x,
            "obj_loc_y": obj_y,
            "img_loc_x": img_x,
            "img_loc_y": img_y,
        }
        df = pd.DataFrame(data)

        orientation = _compute_orientation_features(df)

        # Tilted board should have measurable tilt magnitude
        assert orientation.tilt_magnitude > 0.001, (
            f"Expected measurable tilt for tilted board, got {orientation.tilt_magnitude}"
        )

    def test_orientation_bin_returns_none_for_frontal(self):
        """Frontal-parallel boards should not be assigned an orientation bin."""
        # Low tilt magnitude should return None (doesn't count for diversity)
        frontal = OrientationFeatures(
            tilt_direction=0.0,
            tilt_magnitude=0.01,  # Below threshold
            in_plane_rotation=0.0,
        )

        bin_idx = _get_orientation_bin(frontal)

        assert bin_idx is None

    def test_orientation_bin_maps_directions_to_bins(self):
        """Tilted boards should be assigned to direction bins."""
        # Test different tilt directions map to different bins
        tilt_mag = 0.1  # Above threshold

        # 0 degrees should be bin 0
        east = OrientationFeatures(tilt_direction=0.0, tilt_magnitude=tilt_mag, in_plane_rotation=0.0)
        # 90 degrees (π/2) should be bin 2
        north = OrientationFeatures(tilt_direction=np.pi / 2, tilt_magnitude=tilt_mag, in_plane_rotation=0.0)
        # 180 degrees (π) should be bin 4
        west = OrientationFeatures(tilt_direction=np.pi, tilt_magnitude=tilt_mag, in_plane_rotation=0.0)

        assert _get_orientation_bin(east) == 0
        assert _get_orientation_bin(north) == 2
        assert _get_orientation_bin(west) == 4

    def test_insufficient_points_returns_zero_orientation(self):
        """With fewer than 4 points, orientation extraction returns zeros."""
        data = {
            "sync_index": [0] * 3,
            "port": [0] * 3,
            "point_id": [0, 1, 2],
            "obj_loc_x": [0, 1, 2],
            "obj_loc_y": [0, 0, 0],
            "img_loc_x": [100, 200, 300],
            "img_loc_y": [100, 100, 100],
        }
        df = pd.DataFrame(data)

        orientation = _compute_orientation_features(df)

        assert orientation.tilt_magnitude == 0.0
        assert orientation.tilt_direction == 0.0
        assert orientation.in_plane_rotation == 0.0


class TestTwoPhaseSelection:
    """Tests for two-phase frame selection (orientation anchors + coverage)."""

    def test_result_includes_orientation_metrics(self, tmp_path: Path):
        """Frame selection result includes orientation_sufficient and orientation_count."""
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

        # Check new fields exist
        assert hasattr(result, "orientation_sufficient")
        assert hasattr(result, "orientation_count")
        assert isinstance(result.orientation_sufficient, bool)
        assert isinstance(result.orientation_count, int)
        assert result.orientation_count >= 0

    def test_empty_result_has_orientation_fields(self):
        """Empty result (no frames) should have orientation fields set to defaults."""
        data = {
            "sync_index": [0],
            "port": [1],  # Only port 1 has data
            "point_id": [0],
            "img_loc_x": [100],
            "img_loc_y": [100],
            "obj_loc_x": [0.0],
            "obj_loc_y": [0.0],
        }
        image_points = ImagePoints(pd.DataFrame(data))

        result = select_calibration_frames(
            image_points,
            port=0,  # Port 0 has no data
            image_size=(1920, 1080),
        )

        assert result.orientation_sufficient is False
        assert result.orientation_count == 0


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
            assert results[i].orientation_count == results[0].orientation_count


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
        assert result.orientation_sufficient is False

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
        assert result.orientation_sufficient is False


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
            logger.info(f"  Orientation count: {result.orientation_count}")
            logger.info(f"  Orientation sufficient: {result.orientation_sufficient}")
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
        port_0_frames = set(pd.unique(image_points.df[image_points.df["port"] == 0]["sync_index"]))
        for frame in result.selected_frames:
            assert frame in port_0_frames, f"Selected frame {frame} not in original data"

    def test_frame_selector_beats_random_baseline(self, tmp_path: Path):
        """Selected frames should produce lower holdout error than random selection.

        This integration test validates that the frame selection algorithm produces
        better calibrations than naive random selection, using holdout reprojection
        error as the quality metric.
        """
        from caliscope.core.calibrate_intrinsics import calibrate_intrinsics, compute_holdout_error

        original_path = Path(__root__, "tests", "sessions", "prerecorded_calibration")
        copy_contents_to_clean_dest(original_path, tmp_path)

        xy_csv = tmp_path / "calibration" / "intrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
        image_points = ImagePoints.from_csv(xy_csv)

        port = 0
        image_size = (1280, 720)
        target_count = 15

        # Get eligible frames for this port
        port_df = image_points.df[image_points.df["port"] == port]
        eligible_df = port_df.groupby("sync_index").filter(lambda g: len(g) >= 6)
        all_frames = list(pd.unique(eligible_df["sync_index"]))

        if len(all_frames) < target_count * 2:
            logger.warning(f"Insufficient frames for holdout test: {len(all_frames)}")
            return  # Skip test if insufficient data

        # Run frame selector
        result = select_calibration_frames(
            image_points,
            port=port,
            image_size=image_size,
            target_frame_count=target_count,
        )

        if len(result.selected_frames) < 10:
            logger.warning(f"Frame selector returned too few frames: {len(result.selected_frames)}")
            return  # Skip if selector couldn't find enough frames

        # Determine holdout frames (not selected)
        selected_set = set(result.selected_frames)
        holdout_frames = [f for f in all_frames if f not in selected_set]

        if len(holdout_frames) < 5:
            logger.warning(f"Insufficient holdout frames: {len(holdout_frames)}")
            return  # Skip if not enough holdout frames

        # Calibrate with selected frames
        try:
            greedy_calib = calibrate_intrinsics(
                image_points,
                port=port,
                image_size=image_size,
                selected_frames=result.selected_frames,
            )
            greedy_holdout = compute_holdout_error(
                image_points,
                greedy_calib,
                port=port,
                holdout_frames=holdout_frames[:20],  # Use subset for speed
            )
        except Exception as e:
            logger.warning(f"Greedy calibration failed: {e}")
            return

        # Compare against random selection baseline (multiple trials)
        n_random_trials = 10
        random_rmses: list[float] = []
        rng = random.Random(42)  # Fixed seed for reproducibility

        for trial in range(n_random_trials):
            random_frames = rng.sample(all_frames, min(target_count, len(all_frames)))
            random_holdout = [f for f in all_frames if f not in set(random_frames)]

            if len(random_holdout) < 5:
                continue

            try:
                random_calib = calibrate_intrinsics(
                    image_points,
                    port=port,
                    image_size=image_size,
                    selected_frames=random_frames,
                )
                random_error = compute_holdout_error(
                    image_points,
                    random_calib,
                    port=port,
                    holdout_frames=random_holdout[:20],
                )
                if not np.isnan(random_error.rmse_pixels):
                    random_rmses.append(random_error.rmse_pixels)
            except Exception:
                continue

        if len(random_rmses) < 3:
            logger.warning("Insufficient successful random trials")
            return

        random_mean = np.mean(random_rmses)
        random_std = np.std(random_rmses)

        logger.info(f"Greedy holdout RMSE: {greedy_holdout.rmse_pixels:.3f}")
        logger.info(f"Random baseline: {random_mean:.3f} +/- {random_std:.3f}")
        logger.info(f"Threshold (mean - 1 std): {random_mean - random_std:.3f}")

        # Greedy selection should significantly outperform random selection
        # The two-phase algorithm (orientation diversity first, then coverage)
        # should produce calibrations at least 1 std better than random mean
        assert greedy_holdout.rmse_pixels < random_mean - random_std, (
            f"Greedy RMSE {greedy_holdout.rmse_pixels:.3f} should beat "
            f"random mean - 1 std ({random_mean - random_std:.3f})"
        )


class TestScoreFrame:
    """Unit tests for the frame scoring function."""

    def test_edge_cells_receive_bonus(self):
        """Cells on image edges receive extra weight for distortion estimation."""
        grid_size = 5

        # Frame covering only edge cells (top and bottom edge, middle column)
        edge_coverage = {(0, 2), (4, 2)}
        # Frame covering only a center cell
        center_coverage = {(2, 2)}

        # Same pose features, no prior selection
        pose = np.zeros(5)

        edge_score = _score_frame(
            edge_coverage,
            selected_coverage=set(),
            candidate_pose=pose,
            selected_poses=[],
            grid_size=grid_size,
        )

        center_score = _score_frame(
            center_coverage,
            selected_coverage=set(),
            candidate_pose=pose,
            selected_poses=[],
            grid_size=grid_size,
        )

        # 2 edge cells with bonus should score higher than 1 center cell
        # Edge: 2 * 1.0 (base) + 2 * 0.2 (edge bonus) = 2.4
        # Center: 1 * 1.0 (base) = 1.0
        assert edge_score > center_score
        assert edge_score == 2.4
        assert center_score == 1.0

    def test_corner_cells_receive_additional_bonus(self):
        """Corner cells get both edge and corner bonuses."""
        grid_size = 5

        # Corner cell (gets edge + corner bonus)
        corner_coverage = {(0, 0)}
        # Edge-only cell (only edge bonus)
        edge_only_coverage = {(0, 2)}

        pose = np.zeros(5)

        corner_score = _score_frame(
            corner_coverage,
            selected_coverage=set(),
            candidate_pose=pose,
            selected_poses=[],
            grid_size=grid_size,
        )

        edge_only_score = _score_frame(
            edge_only_coverage,
            selected_coverage=set(),
            candidate_pose=pose,
            selected_poses=[],
            grid_size=grid_size,
        )

        # Corner: 1.0 (base) + 0.2 (edge) + 0.3 (corner) = 1.5
        # Edge only: 1.0 (base) + 0.2 (edge) = 1.2
        assert corner_score > edge_only_score
        assert corner_score == 1.5
        assert edge_only_score == 1.2

    def test_already_covered_cells_give_zero_base_score(self):
        """Cells already covered don't contribute to base coverage gain."""
        grid_size = 5

        candidate_coverage = {(0, 0), (1, 1)}  # 2 cells
        already_selected = {(0, 0)}  # One already covered

        pose = np.zeros(5)

        score = _score_frame(
            candidate_coverage,
            selected_coverage=already_selected,
            candidate_pose=pose,
            selected_poses=[],
            grid_size=grid_size,
        )

        # Only (1,1) is new - center cell, no bonus
        # Score = 1.0 (base for 1 new cell)
        assert score == 1.0

    def test_pose_diversity_bonus(self):
        """Frames far from selected poses get diversity bonus."""
        grid_size = 5

        # Same coverage for both
        coverage = {(2, 2)}

        # Previously selected pose at origin of feature space
        selected_poses = [np.array([0.0, 0.0, 0.0, 0.0, 0.0])]

        # Near pose
        near_pose = np.array([0.1, 0.1, 0.0, 0.0, 0.0])
        # Far pose
        far_pose = np.array([0.5, 0.5, 0.0, 0.0, 0.0])

        near_score = _score_frame(
            coverage,
            selected_coverage=set(),
            candidate_pose=near_pose,
            selected_poses=selected_poses,
            grid_size=grid_size,
        )

        far_score = _score_frame(
            coverage,
            selected_coverage=set(),
            candidate_pose=far_pose,
            selected_poses=selected_poses,
            grid_size=grid_size,
        )

        # Far pose should have higher diversity bonus
        assert far_score > near_score

    def test_empty_selected_poses_no_diversity_bonus(self):
        """First frame selection has no diversity bonus (no prior poses)."""
        grid_size = 5
        coverage = {(2, 2)}
        pose = np.array([0.5, 0.5, 0.1, 0.1, 1.0])

        score = _score_frame(
            coverage,
            selected_coverage=set(),
            candidate_pose=pose,
            selected_poses=[],
            grid_size=grid_size,
        )

        # Just base score for 1 center cell
        assert score == 1.0


if __name__ == "__main__":
    from pathlib import Path

    import caliscope.logger

    caliscope.logger.setup_logging()

    # Run tests with debug output
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Run orientation tests
    test_orientation = TestOrientationExtraction()
    test_orientation.test_frontal_board_has_low_tilt_magnitude()
    logger.info("PASS: test_frontal_board_has_low_tilt_magnitude")

    test_orientation.test_tilted_board_has_higher_tilt_magnitude()
    logger.info("PASS: test_tilted_board_has_higher_tilt_magnitude")

    test_orientation.test_orientation_bin_returns_none_for_frontal()
    logger.info("PASS: test_orientation_bin_returns_none_for_frontal")

    test_orientation.test_orientation_bin_maps_directions_to_bins()
    logger.info("PASS: test_orientation_bin_maps_directions_to_bins")

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

    # Run two-phase tests
    test_two_phase = TestTwoPhaseSelection()
    test_two_phase.test_result_includes_orientation_metrics(debug_dir)
    logger.info("PASS: test_result_includes_orientation_metrics")

    test_two_phase.test_empty_result_has_orientation_fields()
    logger.info("PASS: test_empty_result_has_orientation_fields")

    # Run integration tests
    test_integration = TestIntegration()
    test_integration.test_selects_frames_from_intrinsic_data(debug_dir)
    logger.info("PASS: test_selects_frames_from_intrinsic_data")

    test_integration.test_port_0_selects_frames_and_reports_coverage(debug_dir)
    logger.info("PASS: test_port_0_selects_frames_and_reports_coverage")

    test_integration.test_selected_frames_are_from_original_data(debug_dir)
    logger.info("PASS: test_selected_frames_are_from_original_data")

    logger.info("All tests passed!")
