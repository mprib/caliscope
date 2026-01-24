"""Tests for SyntheticScene."""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.synthetic import (
    CalibrationObject,
    FilterConfig,
    SyntheticScene,
    Trajectory,
)
from caliscope.synthetic.camera_rigs import ring_rig


class TestConstruction:
    """Test SyntheticScene construction and validation."""

    def test_valid_scene_accepted(self):
        """Create scene with valid inputs."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
            pixel_noise_sigma=0.5,
            random_seed=42,
        )

        assert scene.n_cameras == 4
        assert scene.n_frames == 10

    def test_default_noise_and_seed(self):
        """Default noise sigma is 0.5, seed is 42."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        assert scene.pixel_noise_sigma == 0.5
        assert scene.random_seed == 42

    def test_negative_noise_sigma_rejected(self):
        """Negative pixel noise is invalid."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        with pytest.raises(ValueError, match="pixel_noise_sigma must be >= 0"):
            SyntheticScene(
                camera_array=cameras,
                calibration_object=obj,
                trajectory=traj,
                pixel_noise_sigma=-1.0,
            )

    def test_unposed_cameras_rejected(self):
        """All cameras must have extrinsics."""
        # Create ring rig and strip extrinsics
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        from caliscope.synthetic.camera_rigs import strip_extrinsics

        cameras_no_extrinsics = strip_extrinsics(cameras)

        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        with pytest.raises(ValueError, match="All cameras must have extrinsics"):
            SyntheticScene(
                camera_array=cameras_no_extrinsics,
                calibration_object=obj,
                trajectory=traj,
            )


class TestWorldPoints:
    """Test world_points property."""

    def test_world_points_shape(self):
        """World points has n_frames * n_points rows."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        n_points = obj.n_points  # 5 * 7 = 35
        expected_rows = scene.n_frames * n_points  # 10 * 35 = 350

        assert len(scene.world_points.df) == expected_rows

    def test_world_points_columns(self):
        """World points has expected columns."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        df = scene.world_points.df
        assert "sync_index" in df.columns
        assert "point_id" in df.columns
        assert "x_coord" in df.columns
        assert "y_coord" in df.columns
        assert "z_coord" in df.columns
        assert "frame_time" in df.columns

    def test_world_points_matches_trajectory(self):
        """World points at frame 0 match trajectory transformation."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=5, radius_mm=500.0, origin_frame=0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        # Get world points for frame 0
        frame_0_points = scene.world_points.df[scene.world_points.df["sync_index"] == 0]

        # Manually compute expected world coords
        expected_coords = traj.world_points_at_frame(obj, 0)

        # Compare
        actual_coords = frame_0_points[["x_coord", "y_coord", "z_coord"]].values

        assert np.allclose(actual_coords, expected_coords)

    def test_world_points_all_point_ids_present_per_frame(self):
        """Each frame has all point IDs."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        df = scene.world_points.df

        for frame in range(scene.n_frames):
            frame_points = df[df["sync_index"] == frame]
            point_ids = sorted(frame_points["point_id"].unique())
            expected_ids = sorted(obj.point_ids)

            assert point_ids == expected_ids


class TestImagePoints:
    """Test image_points_perfect and image_points_noisy."""

    def test_image_points_columns(self):
        """Image points have expected columns including obj_loc_x/y/z."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        df = scene.image_points_perfect.df

        expected_cols = [
            "sync_index",
            "port",
            "point_id",
            "img_loc_x",
            "img_loc_y",
            "obj_loc_x",
            "obj_loc_y",
            "obj_loc_z",
            "frame_time",
        ]

        for col in expected_cols:
            assert col in df.columns

    def test_image_points_within_bounds(self):
        """All projected points are within camera bounds."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        df = scene.image_points_noisy.df

        for port, camera in cameras.cameras.items():
            w, h = camera.size
            port_points = df[df["port"] == port]

            assert (port_points["img_loc_x"] >= 0).all()
            assert (port_points["img_loc_x"] < w).all()
            assert (port_points["img_loc_y"] >= 0).all()
            assert (port_points["img_loc_y"] < h).all()

    def test_perfect_vs_noisy_differ_when_noise_nonzero(self):
        """Noisy points differ from perfect when sigma > 0."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
            pixel_noise_sigma=1.0,
        )

        perfect = scene.image_points_perfect.df
        noisy = scene.image_points_noisy.df

        # Merge on key columns to compare same observations
        merged = perfect.merge(
            noisy,
            on=["sync_index", "port", "point_id"],
            suffixes=("_perfect", "_noisy"),
        )

        # Should have some differences
        x_diffs = np.abs(merged["img_loc_x_perfect"] - merged["img_loc_x_noisy"])
        y_diffs = np.abs(merged["img_loc_y_perfect"] - merged["img_loc_y_noisy"])

        # Most points should differ (noise is random)
        assert (x_diffs > 0.01).sum() > len(merged) * 0.9
        assert (y_diffs > 0.01).sum() > len(merged) * 0.9

    def test_perfect_equals_noisy_when_noise_zero(self):
        """Perfect and noisy are identical when sigma = 0."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
            pixel_noise_sigma=0.0,
        )

        perfect = scene.image_points_perfect.df
        noisy = scene.image_points_noisy.df

        # Should be identical
        assert len(perfect) == len(noisy)

        # Merge and compare
        merged = perfect.merge(
            noisy,
            on=["sync_index", "port", "point_id"],
            suffixes=("_perfect", "_noisy"),
        )

        assert np.allclose(merged["img_loc_x_perfect"], merged["img_loc_x_noisy"], atol=1e-10)
        assert np.allclose(merged["img_loc_y_perfect"], merged["img_loc_y_noisy"], atol=1e-10)

    def test_obj_loc_columns_match_calibration_object(self):
        """obj_loc_x/y/z match the calibration object's local coordinates."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=3, cols=4, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=5, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        df = scene.image_points_perfect.df

        # Check each point_id
        for point_id in obj.point_ids:
            point_rows = df[df["point_id"] == point_id]

            # All observations of this point should have same obj_loc
            obj_locs = point_rows[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].values

            # Should be constant across all observations
            assert np.allclose(obj_locs, obj_locs[0], atol=1e-10)

            # Should match the object's local coordinates
            idx = np.where(obj.point_ids == point_id)[0][0]
            expected = obj.points[idx]

            assert np.allclose(obj_locs[0], expected, atol=1e-10)

    def test_random_seed_controls_noise(self):
        """Different random seeds produce different noise patterns."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene1 = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
            pixel_noise_sigma=1.0,
            random_seed=42,
        )

        scene2 = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
            pixel_noise_sigma=1.0,
            random_seed=99,
        )

        noisy1 = scene1.image_points_noisy.df
        noisy2 = scene2.image_points_noisy.df

        # Merge on key columns
        merged = noisy1.merge(noisy2, on=["sync_index", "port", "point_id"], suffixes=("_1", "_2"))

        # Should be different
        x_same = np.isclose(merged["img_loc_x_1"], merged["img_loc_x_2"], atol=1e-10)
        y_same = np.isclose(merged["img_loc_y_1"], merged["img_loc_y_2"], atol=1e-10)

        # Most should differ
        assert x_same.sum() < len(merged) * 0.1
        assert y_same.sum() < len(merged) * 0.1

    def test_same_seed_produces_reproducible_noise(self):
        """Same random seed produces identical noise."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene1 = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
            pixel_noise_sigma=1.0,
            random_seed=42,
        )

        scene2 = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
            pixel_noise_sigma=1.0,
            random_seed=42,
        )

        noisy1 = scene1.image_points_noisy.df.sort_values(["sync_index", "port", "point_id"])
        noisy2 = scene2.image_points_noisy.df.sort_values(["sync_index", "port", "point_id"])

        # Should be identical
        assert np.allclose(noisy1["img_loc_x"], noisy2["img_loc_x"], atol=1e-10)
        assert np.allclose(noisy1["img_loc_y"], noisy2["img_loc_y"], atol=1e-10)


class TestCoverageMatrix:
    """Test coverage_matrix property."""

    def test_coverage_matrix_shape(self):
        """Coverage matrix is (n_cameras, n_cameras)."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        assert scene.coverage_matrix.shape == (4, 4)

    def test_coverage_matrix_symmetric(self):
        """Coverage matrix is symmetric."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        cov = scene.coverage_matrix

        assert np.allclose(cov, cov.T)

    def test_coverage_matrix_diagonal_is_total_observations(self):
        """Diagonal elements are total observations per camera."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        cov = scene.coverage_matrix
        df = scene.image_points_noisy.df

        for port in range(scene.n_cameras):
            port_count = len(df[df["port"] == port])
            assert cov[port, port] == port_count

    def test_coverage_matrix_off_diagonal_is_shared_observations(self):
        """Off-diagonal elements count shared observations."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        cov = scene.coverage_matrix
        df = scene.image_points_noisy.df

        # Check one pair
        port_a = 0
        port_b = 1

        # Find shared (sync_index, point_id) pairs
        a_obs = set(
            zip(
                df[df["port"] == port_a]["sync_index"],
                df[df["port"] == port_a]["point_id"],
            )
        )
        b_obs = set(
            zip(
                df[df["port"] == port_b]["sync_index"],
                df[df["port"] == port_b]["point_id"],
            )
        )

        shared_count = len(a_obs & b_obs)

        assert cov[port_a, port_b] == shared_count
        assert cov[port_b, port_a] == shared_count


class TestIntrinsicsOnlyCameras:
    """Test intrinsics_only_cameras method."""

    def test_strips_extrinsics(self):
        """Returned cameras have no extrinsics."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        intrinsics_only = scene.intrinsics_only_cameras()

        for port, camera in intrinsics_only.cameras.items():
            assert camera.rotation is None
            assert camera.translation is None

    def test_preserves_intrinsics(self):
        """Returned cameras have same intrinsics."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        intrinsics_only = scene.intrinsics_only_cameras()

        for port, camera in scene.camera_array.cameras.items():
            intrinsics_cam = intrinsics_only.cameras[port]

            assert np.allclose(camera.matrix, intrinsics_cam.matrix)
            assert np.allclose(camera.distortions, intrinsics_cam.distortions)
            assert camera.size == intrinsics_cam.size


class TestApplyFilter:
    """Test apply_filter method."""

    def test_apply_empty_filter_returns_unchanged(self):
        """Empty filter returns same points."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        config = FilterConfig()
        filtered = scene.apply_filter(config)

        assert len(filtered.df) == len(scene.image_points_noisy.df)

    def test_apply_dropped_cameras_filter(self):
        """Dropped cameras filter removes observations."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        config = FilterConfig(dropped_cameras=(0, 2))
        filtered = scene.apply_filter(config)

        # Should have no observations from cameras 0 or 2
        assert 0 not in filtered.df["port"].values
        assert 2 not in filtered.df["port"].values
        assert 1 in filtered.df["port"].values
        assert 3 in filtered.df["port"].values

    def test_apply_killed_linkage_filter(self):
        """Killed linkage filter removes shared observations."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        # Before filtering
        df_before = scene.image_points_noisy.df
        cam0_obs = set(
            zip(
                df_before[df_before["port"] == 0]["sync_index"],
                df_before[df_before["port"] == 0]["point_id"],
            )
        )
        cam1_obs = set(
            zip(
                df_before[df_before["port"] == 1]["sync_index"],
                df_before[df_before["port"] == 1]["point_id"],
            )
        )
        shared_before = len(cam0_obs & cam1_obs)

        # Apply filter
        config = FilterConfig(killed_linkages=((0, 1),))
        filtered = scene.apply_filter(config)

        # After filtering
        df_after = filtered.df
        cam0_obs_after = set(
            zip(
                df_after[df_after["port"] == 0]["sync_index"],
                df_after[df_after["port"] == 0]["point_id"],
            )
        )
        cam1_obs_after = set(
            zip(
                df_after[df_after["port"] == 1]["sync_index"],
                df_after[df_after["port"] == 1]["point_id"],
            )
        )
        shared_after = len(cam0_obs_after & cam1_obs_after)

        # Should have no shared observations
        assert shared_before > 0  # There were shared observations before
        assert shared_after == 0  # No shared observations after

    def test_apply_random_dropout_reduces_observations(self):
        """Random dropout reduces number of observations."""
        cameras = ring_rig(n_cameras=4, radius_mm=2000.0)
        obj = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)
        traj = Trajectory.orbital(n_frames=10, radius_mm=500.0)

        scene = SyntheticScene(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=traj,
        )

        n_before = len(scene.image_points_noisy.df)

        config = FilterConfig(random_dropout_fraction=0.5)
        filtered = scene.apply_filter(config)

        n_after = len(filtered.df)

        # Should have roughly half the points (with some variance)
        assert n_after < n_before
        assert 0.4 * n_before < n_after < 0.6 * n_before


if __name__ == "__main__":
    """Debug harness for running tests with saved outputs."""
    import sys

    # Run pytest on this file
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
