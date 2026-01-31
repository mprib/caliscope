"""Tests for bundle adjustment with unposed cameras.

Replaces slow integration tests that loaded 210K-line CSV files.
Target runtime: <2 seconds total.
"""

import pytest

from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.core.point_data import ImagePoints
from caliscope.core.point_data_bundle import PointDataBundle
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.filter_config import FilterConfig
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.trajectory import Trajectory


def make_12_camera_scene() -> SyntheticScene:
    """Create 12-camera scene for unposed camera tests.

    Configuration:
    - 12 cameras (ports 0-11) arranged in a ring
    - 2000mm radius, 500mm height
    - 5x7 grid at 50mm spacing (250x300mm total)
    - 5 stationary frames (sufficient for crash testing)
    """
    camera_array = CameraSynthesizer().add_ring(n=12, radius_mm=2000.0, height_mm=500.0).build()
    calibration_object = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
    trajectory = Trajectory.stationary(n_frames=5)

    return SyntheticScene(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=0.5,
    )


class TestDroppedCameras:
    """Test bundle adjustment when cameras have zero observations."""

    def test_optimization_with_dropped_cameras(self) -> None:
        """Cameras with zero observations are excluded from optimization.

        Scenario: Cameras 9 and 10 have no observations in image_points.
        Expected: Only cameras 0-8 and 11 are posed; optimization converges.
        """
        scene = make_12_camera_scene()

        # Drop cameras 9 and 10 completely
        config = FilterConfig(dropped_cameras=(9, 10))
        filtered_image_points = scene.apply_filter(config)

        # Bootstrap uses intrinsics-only cameras
        intrinsics_only = scene.intrinsics_only_cameras()

        # Build pose network (will not find cameras 9, 10)
        network = build_paired_pose_network(filtered_image_points, intrinsics_only, method="stereocalibrate")
        network.apply_to(intrinsics_only)

        # Triangulate with posed cameras only
        world_points = filtered_image_points.triangulate(intrinsics_only)

        # Create bundle and verify setup
        bundle = PointDataBundle(intrinsics_only, filtered_image_points, world_points)

        # Key assertions on camera configuration
        posed_ports = set(intrinsics_only.posed_cameras.keys())
        unposed_ports = set(intrinsics_only.unposed_cameras.keys())

        expected_posed = {0, 1, 2, 3, 4, 5, 6, 7, 8, 11}
        assert posed_ports == expected_posed, f"Expected ports {expected_posed} posed, got {posed_ports}"
        assert unposed_ports == {9, 10}, f"Expected ports 9,10 unposed, got {unposed_ports}"
        assert len(intrinsics_only.posed_port_to_index) == 10

        # Core test: optimization completes without crashing
        optimized = bundle.optimize()

        assert optimized.optimization_status is not None
        assert optimized.optimization_status.converged
        assert optimized.reprojection_report.overall_rmse < 5.0  # Sanity check


class TestIsolatedIslands:
    """Test bundle adjustment when cameras form two disconnected islands.

    This is an EXPLORATORY test. We don't know what the expected behavior is
    when two equal-sized camera islands exist with no shared observations.
    The test documents whatever behavior is discovered.
    """

    def test_optimization_with_two_isolated_islands(self) -> None:
        """Two 6-camera islands with no shared observations between them.

        Scenario:
        - Island A: cameras 0-5 see only frames 0-2
        - Island B: cameras 6-11 see only frames 3-4
        - NO shared observations between any camera in A and any camera in B
          (because they observe different frames)

        Expected behavior is UNKNOWN. This test documents what actually happens:
        - Does one island become "posed" and the other "unposed"?
        - If so, which island wins? (Is it deterministic?)
        - Does optimization converge with only one island?
        - Or does the entire calibration fail?
        """
        scene = make_12_camera_scene()

        # Define islands
        island_a = set(range(6))  # {0, 1, 2, 3, 4, 5}
        island_b = set(range(6, 12))  # {6, 7, 8, 9, 10, 11}

        # Define frame visibility per island
        frames_a = {0, 1, 2}  # Island A sees these frames
        frames_b = {3, 4}  # Island B sees these frames

        # Get raw image points from scene
        image_points_df = scene.image_points_noisy.df.copy()

        # Filter: keep observations where (camera in island_a AND frame in frames_a)
        #                              OR (camera in island_b AND frame in frames_b)
        mask_island_a = (image_points_df["port"].isin(island_a)) & (image_points_df["sync_index"].isin(frames_a))
        mask_island_b = (image_points_df["port"].isin(island_b)) & (image_points_df["sync_index"].isin(frames_b))

        filtered_df = image_points_df[mask_island_a | mask_island_b].copy()
        filtered_image_points = ImagePoints(filtered_df)

        # Verify both islands still have internal observations
        for port in range(12):
            port_obs = filtered_image_points.df[filtered_image_points.df["port"] == port]
            assert len(port_obs) > 0, f"Camera {port} should still have observations"

        # Bootstrap uses intrinsics-only cameras
        intrinsics_only = scene.intrinsics_only_cameras()

        # Build pose network - this is where the interesting behavior happens
        network = build_paired_pose_network(filtered_image_points, intrinsics_only, method="stereocalibrate")
        network.apply_to(intrinsics_only)

        # Document what happened
        posed_ports = set(intrinsics_only.posed_cameras.keys())
        unposed_ports = set(intrinsics_only.unposed_cameras.keys())

        print("\n=== DISCOVERED BEHAVIOR ===")
        print(f"Posed ports: {sorted(posed_ports)}")
        print(f"Unposed ports: {sorted(unposed_ports)}")
        print(f"Number posed: {len(posed_ports)}")
        print(f"Number unposed: {len(unposed_ports)}")

        # Document which island was chosen (if any)
        posed_in_island_a = posed_ports & island_a
        posed_in_island_b = posed_ports & island_b
        print(f"Posed from Island A (0-5): {sorted(posed_in_island_a)}")
        print(f"Posed from Island B (6-11): {sorted(posed_in_island_b)}")

        # Assertions that MUST be true regardless of which island wins
        # (These document the invariants we expect)

        # At least some cameras should be posed (calibration shouldn't completely fail)
        assert len(posed_ports) > 0, "Expected at least some cameras to be posed"

        # The posed cameras should form a connected component
        # (i.e., we shouldn't have cameras from both islands posed,
        # since they have no shared observations)
        both_islands_posed = len(posed_in_island_a) > 0 and len(posed_in_island_b) > 0
        if both_islands_posed:
            print("WARNING: Cameras from BOTH islands are posed. This is unexpected!")
            print("  This suggests the system is not correctly detecting disconnected components.")

        # If only one island is posed, verify it's a complete island
        if not both_islands_posed:
            if len(posed_in_island_a) > 0:
                assert posed_in_island_a == island_a, f"Expected all of Island A to be posed, got {posed_in_island_a}"
                assert posed_in_island_b == set(), f"Expected no cameras from Island B, got {posed_in_island_b}"
            else:
                assert posed_in_island_b == island_b, f"Expected all of Island B to be posed, got {posed_in_island_b}"
                assert posed_in_island_a == set(), f"Expected no cameras from Island A, got {posed_in_island_a}"

        # Attempt triangulation and optimization (may or may not work)
        if len(posed_ports) >= 2:
            world_points = filtered_image_points.triangulate(intrinsics_only)
            bundle = PointDataBundle(intrinsics_only, filtered_image_points, world_points)

            # Core test: does optimization complete?
            optimized = bundle.optimize()

            print(f"Optimization converged: {optimized.optimization_status.converged}")
            print(f"Reprojection RMSE: {optimized.reprojection_report.overall_rmse:.3f}")

            assert optimized.optimization_status is not None
            # Note: We assert convergence because even with one island,
            # 6 cameras should be enough for stable optimization
            assert optimized.optimization_status.converged
            assert optimized.reprojection_report.overall_rmse < 5.0
        else:
            pytest.skip("Fewer than 2 cameras posed - cannot triangulate")

    def _run_island_test(self, island_a_ports: set[int], island_b_ports: set[int]) -> tuple[set[int], set[int]]:
        """Helper: run island split and return (posed_ports, unposed_ports)."""
        scene = make_12_camera_scene()

        frames_a = {0, 1, 2}
        frames_b = {3, 4}

        image_points_df = scene.image_points_noisy.df.copy()

        mask_a = (image_points_df["port"].isin(island_a_ports)) & (image_points_df["sync_index"].isin(frames_a))
        mask_b = (image_points_df["port"].isin(island_b_ports)) & (image_points_df["sync_index"].isin(frames_b))

        filtered_df = image_points_df[mask_a | mask_b].copy()
        filtered_image_points = ImagePoints(filtered_df)

        intrinsics_only = scene.intrinsics_only_cameras()
        network = build_paired_pose_network(filtered_image_points, intrinsics_only, method="stereocalibrate")
        network.apply_to(intrinsics_only)

        posed_ports = set(intrinsics_only.posed_cameras.keys())
        unposed_ports = set(intrinsics_only.unposed_cameras.keys())

        return posed_ports, unposed_ports

    def test_largest_island_wins_when_smaller_has_port_0(self) -> None:
        """Larger island is chosen even when smaller island contains port 0.

        Scenario: 4-camera island (0-3) vs 8-camera island (4-11)
        Expected: 8-camera island wins despite port 0 being in smaller island.
        """
        island_a = {0, 1, 2, 3}  # 4 cameras, has port 0
        island_b = {4, 5, 6, 7, 8, 9, 10, 11}  # 8 cameras

        posed_ports, unposed_ports = self._run_island_test(island_a, island_b)

        # Larger island (B) should win
        assert posed_ports == island_b, f"Expected larger island {island_b} posed, got {posed_ports}"
        assert unposed_ports == island_a, f"Expected smaller island {island_a} unposed, got {unposed_ports}"

    def test_largest_island_wins_when_larger_has_port_0(self) -> None:
        """Larger island is chosen when it also contains port 0.

        Scenario: 8-camera island (0-7) vs 4-camera island (8-11)
        Expected: 8-camera island wins.
        """
        island_a = {0, 1, 2, 3, 4, 5, 6, 7}  # 8 cameras, has port 0
        island_b = {8, 9, 10, 11}  # 4 cameras

        posed_ports, unposed_ports = self._run_island_test(island_a, island_b)

        # Larger island (A) should win
        assert posed_ports == island_a, f"Expected larger island {island_a} posed, got {posed_ports}"
        assert unposed_ports == island_b, f"Expected smaller island {island_b} unposed, got {unposed_ports}"

    def test_largest_island_wins_non_contiguous_ports(self) -> None:
        """Larger island wins even with non-contiguous port numbers.

        Scenario: 4-camera island (4-7) vs 8-camera island (0-3, 8-11)
        Expected: 8-camera island wins.

        This tests that size, not port ordering, determines the winner.
        """
        island_a = {4, 5, 6, 7}  # 4 cameras, no port 0
        island_b = {0, 1, 2, 3, 8, 9, 10, 11}  # 8 cameras, has port 0

        posed_ports, unposed_ports = self._run_island_test(island_a, island_b)

        # Larger island (B) should win
        assert posed_ports == island_b, f"Expected larger island {island_b} posed, got {posed_ports}"
        assert unposed_ports == island_a, f"Expected smaller island {island_a} unposed, got {unposed_ports}"


def explore_island_split(island_a_ports: set[int], island_b_ports: set[int], label: str) -> None:
    """Explore what happens with a given island split."""
    scene = make_12_camera_scene()

    # Island A sees frames 0-2, Island B sees frames 3-4
    frames_a = {0, 1, 2}
    frames_b = {3, 4}

    image_points_df = scene.image_points_noisy.df.copy()

    mask_a = (image_points_df["port"].isin(island_a_ports)) & (image_points_df["sync_index"].isin(frames_a))
    mask_b = (image_points_df["port"].isin(island_b_ports)) & (image_points_df["sync_index"].isin(frames_b))

    filtered_df = image_points_df[mask_a | mask_b].copy()
    filtered_image_points = ImagePoints(filtered_df)

    intrinsics_only = scene.intrinsics_only_cameras()
    network = build_paired_pose_network(filtered_image_points, intrinsics_only, method="stereocalibrate")
    network.apply_to(intrinsics_only)

    posed_ports = set(intrinsics_only.posed_cameras.keys())
    posed_in_a = posed_ports & island_a_ports
    posed_in_b = posed_ports & island_b_ports

    print(f"\n=== {label} ===")
    print(f"Island A: {sorted(island_a_ports)} ({len(island_a_ports)} cameras)")
    print(f"Island B: {sorted(island_b_ports)} ({len(island_b_ports)} cameras)")
    print("WINNER: ", end="")
    if posed_in_a and not posed_in_b:
        print(f"Island A ({len(posed_in_a)} cameras posed)")
    elif posed_in_b and not posed_in_a:
        print(f"Island B ({len(posed_in_b)} cameras posed)")
    else:
        print(f"BOTH? A={len(posed_in_a)}, B={len(posed_in_b)}")


if __name__ == "__main__":
    """Debug harness for manual testing."""
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing dropped cameras scenario...")
    test = TestDroppedCameras()
    test.test_optimization_with_dropped_cameras()
    print("  PASSED")

    print("\nTesting isolated islands scenario...")
    test = TestIsolatedIslands()
    test.test_optimization_with_two_isolated_islands()
    print("  PASSED")

    # Explore asymmetric splits
    print("\n" + "=" * 60)
    print("EXPLORING ISLAND SELECTION BEHAVIOR")
    print("=" * 60)

    # 6/6 split (baseline)
    explore_island_split(
        island_a_ports={0, 1, 2, 3, 4, 5}, island_b_ports={6, 7, 8, 9, 10, 11}, label="6/6 SPLIT (baseline)"
    )

    # 4/8 split - smaller island has port 0
    explore_island_split(
        island_a_ports={0, 1, 2, 3},
        island_b_ports={4, 5, 6, 7, 8, 9, 10, 11},
        label="4/8 SPLIT - Port 0 in SMALLER island",
    )

    # 8/4 split - larger island has port 0
    explore_island_split(
        island_a_ports={0, 1, 2, 3, 4, 5, 6, 7},
        island_b_ports={8, 9, 10, 11},
        label="8/4 SPLIT - Port 0 in LARGER island",
    )

    # Tricky: 4/8 but port 0 is in the larger island B
    explore_island_split(
        island_a_ports={4, 5, 6, 7},
        island_b_ports={0, 1, 2, 3, 8, 9, 10, 11},
        label="4/8 SPLIT - Port 0 in LARGER island (non-contiguous)",
    )
