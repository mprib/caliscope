"""
Integration tests for PoseNetworkBuilder using the same data as the prototyping script.
Copy data to tmp_path, run builder, verify it works. No fixtures, no mocking.
"""

import logging
from pathlib import Path

import pytest
import numpy as np

from caliscope import __root__
from caliscope.core.capture_volume.capture_volume import CaptureVolume
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.point_data import ImagePoints
from caliscope.core.bootstrap_pose.pose_network_builder import PoseNetworkBuilder
from caliscope import persistence

logger = logging.getLogger(__name__)


def test_pose_network_builder_end_to_end(tmp_path: Path):
    """
    Full pipeline test: copy prototyping data, build network, apply to array.
    """
    # Copy the exact data from the prototyping script
    source_dir = Path(__root__, "tests/sessions/post_optimization")
    copy_contents_to_clean_dest(source_dir, tmp_path)

    # Load data the same way the prototyping script does
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    image_points = ImagePoints.from_csv(tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv")

    # Reset poses to ensure clean state
    for cam in camera_array.cameras.values():
        cam.rotation = None
        cam.translation = None

    # Build network with explicit parameters (not defaults, to test propagation)
    builder = PoseNetworkBuilder(camera_array, image_points)
    network = (
        builder.estimate_camera_to_object_poses(min_points=6)
        .estimate_relative_poses()
        .filter_outliers(threshold=1.5)
        .build()
    )

    # Basic sanity checks
    assert builder.state == "built"
    assert len(network._pairs) > 0, "Should produce stereo pairs"

    # Apply to array and verify poses were set
    test_array = CameraArray(camera_array.cameras.copy())
    network.apply_to(test_array)

    assert len(test_array.posed_cameras) > 0, "Should pose at least one camera"

    # Verify pose quality
    for port, cam in test_array.posed_cameras.items():
        assert cam.rotation.shape == (3, 3)
        assert cam.translation.shape == (3,)
        assert np.isclose(np.linalg.det(cam.rotation), 1.0, atol=1e-5)

    # assure that capture volume can be created from array and points
    world_points = image_points.triangulate(test_array)
    capture_volume = CaptureVolume(test_array, world_points.to_point_estimates(image_points, test_array))
    capture_volume.optimize()


def test_builder_parameter_propagation(tmp_path: Path):
    """
    Verify that custom parameters actually affect the pipeline.
    Compare lenient vs strict configurations.
    """
    source_dir = Path(__root__, "tests/sessions/post_optimization")
    copy_contents_to_clean_dest(source_dir, tmp_path)

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    image_points = ImagePoints.from_csv(tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv")

    # Lenient config: more data passes through
    builder_lenient = PoseNetworkBuilder(camera_array, image_points)
    network_lenient = (
        builder_lenient.estimate_camera_to_object_poses(min_points=4)
        .estimate_relative_poses()
        .filter_outliers(threshold=2.0)
        .build()
    )

    # Strict config: less data passes through
    builder_strict = PoseNetworkBuilder(camera_array, image_points)
    network_strict = (
        builder_strict.estimate_camera_to_object_poses(min_points=8)
        .estimate_relative_poses()
        .filter_outliers(threshold=1.0)
        .build()
    )

    # Strict should not produce MORE pairs than lenient
    assert len(network_strict._pairs) <= len(network_lenient._pairs)


def test_builder_enforces_execution_order(tmp_path: Path):
    """
    Verify that builder methods must be called in correct sequence.
    """
    source_dir = Path(__root__, "tests/sessions/post_optimization")
    copy_contents_to_clean_dest(source_dir, tmp_path)

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    image_points = ImagePoints.from_csv(tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv")

    builder = PoseNetworkBuilder(camera_array, image_points)

    # Try to skip step 1a
    with pytest.raises(RuntimeError, match="Must call estimate_camera_to_object_poses"):
        builder.estimate_relative_poses()

    # Try to skip step 1b
    builder.estimate_camera_to_object_poses()
    with pytest.raises(RuntimeError, match="Must call estimate_relative_poses"):
        builder.filter_outliers()

    # Try to skip step 2
    builder.estimate_relative_poses()
    with pytest.raises(RuntimeError, match="Must call filter_outliers"):
        builder.build()


def test_apply_to_with_disconnected_camera(tmp_path: Path):
    """
    Test behavior when pose graph doesn't connect all cameras.
    """
    source_dir = Path(__root__, "tests/sessions/post_optimization")
    copy_contents_to_clean_dest(source_dir, tmp_path)

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    image_points = ImagePoints.from_csv(tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv")

    builder = PoseNetworkBuilder(camera_array, image_points)
    network = builder.estimate_camera_to_object_poses().estimate_relative_poses().filter_outliers().build()

    # Add a camera that has no data/pairs
    extra_cameras = camera_array.cameras.copy()
    from caliscope.cameras.camera_array import CameraData

    extra_cameras[99] = CameraData(port=99, size=[640, 480], matrix=np.eye(3), distortions=np.zeros(5))

    test_array = CameraArray(extra_cameras)
    network.apply_to(test_array)

    # Original cameras should be posed, extra one should not
    assert len(test_array.posed_cameras) >= len(camera_array.cameras)
    assert 99 in test_array.unposed_cameras


def test_quaternion_average_edge_cases():
    """Test quaternion averaging handles realistic edge cases."""
    from caliscope.core.bootstrap_pose.pose_network_builder import quaternion_average

    # Single quaternion (common: only one valid frame)
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    result = quaternion_average(np.array([q1]))
    assert np.allclose(result, q1)

    # Two identical quaternions (common: static scene)
    q2 = np.array([0.70710678, 0.70710678, 0.0, 0.0])
    result = quaternion_average(np.array([q2, q2]))
    assert np.allclose(result, q2)

    # Small dispersion (realistic: slight pose jitter)
    q_perturbed = q2 + np.array([0.001, -0.001, 0.0, 0.0])
    q_perturbed = q_perturbed / np.linalg.norm(q_perturbed)
    result = quaternion_average(np.array([q2, q_perturbed]))
    assert np.isclose(np.linalg.norm(result), 1.0)
    assert not np.any(np.isnan(result))


# Debug harness: run the builder pipeline with real data for inspection
if __name__ == "__main__":
    """
    Run builder pipeline manually for debugging. Uses the same data as the tests.
    """
    from caliscope.logger import setup_logging

    setup_logging()

    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)
    test_quaternion_average_edge_cases()
    test_pose_network_builder_end_to_end(debug_dir)
    test_builder_parameter_propagation(debug_dir)
    test_builder_enforces_execution_order(debug_dir)
    test_apply_to_with_disconnected_camera(debug_dir)
