import logging
from pathlib import Path

import numpy as np
import pandas as pd

from caliscope import __root__
from caliscope import persistence
from caliscope.calibration.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.calibration.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray, CameraData

logger = logging.getLogger(__name__)


def create_test_camera_array() -> CameraArray:
    """Create a realistic CameraArray with mixed intrinsic-only and fully calibrated cameras."""
    cameras = {
        0: CameraData(
            port=0,
            size=[1920, 1080],
            rotation_count=0,
            error=0.5,
            matrix=np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float64),
            distortions=np.array([0.1, -0.2, 0, 0, 0.3], dtype=np.float64),
            exposure=100,
            grid_count=20,
            ignore=False,
            translation=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            fisheye=False,
        ),
        1: CameraData(
            port=1,
            size=[1280, 720],
            rotation_count=1,
            error=0.7,
            matrix=np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64),
            distortions=np.array([0.05, -0.1, 0, 0, 0.15], dtype=np.float64),
            exposure=120,
            grid_count=15,
            ignore=False,
            translation=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
            fisheye=False,
        ),
        2: CameraData(
            port=2,
            size=[640, 480],
            rotation_count=0,
            error=None,  # Intrinsic-only camera
            matrix=np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float64),
            distortions=np.array([0.01, -0.02, 0, 0, 0.03], dtype=np.float64),
            exposure=80,
            grid_count=10,
            ignore=False,
            translation=None,  # No extrinsics
            rotation=None,  # No extrinsics
            fisheye=True,
        ),
    }
    return CameraArray(cameras)


def create_test_charuco() -> Charuco:
    """Create a realistic Charuco board definition."""
    return Charuco(
        columns=4,
        rows=5,
        board_height=8.5,
        board_width=11,
        dictionary="DICT_4X4_50",
        units="inch",
        aruco_scale=0.75,
        square_size_overide_cm=5.4,
        inverted=False,
        legacy_pattern=False,
    )


def load_fixture_data(session_name: str = "post_optimization") -> dict:
    """Load real fixture data from tests/sessions directory."""
    session_path = Path(__root__, "tests", "sessions", session_name)

    # Load camera array
    camera_array_path = session_path / "camera_array.toml"
    camera_array = persistence.load_camera_array(camera_array_path)

    # Load charuco
    charuco_path = session_path / "charuco.toml"
    charuco = persistence.load_charuco(charuco_path)

    # Load point estimates
    point_estimates_path = session_path / "point_estimates.toml"
    point_estimates = persistence.load_point_estimates(point_estimates_path)

    # Load image points
    xy_csv_path = session_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
    image_points = persistence.load_image_points_csv(xy_csv_path)

    # Build paired pose network
    paired_pose_network = build_paired_pose_network(image_points, camera_array)

    return {
        "camera_array": camera_array,
        "charuco": charuco,
        "point_estimates": point_estimates,
        "image_points": image_points,
        "paired_pose_network": paired_pose_network,
    }


def test_camera_array_roundtrip(tmp_path: Path):
    """Test CameraArray save/load round-trip."""
    logger.info("Testing CameraArray round-trip...")

    # Create test data
    original = create_test_camera_array()
    file_path = tmp_path / "camera_array.toml"

    # Save and load
    persistence.save_camera_array(original, file_path)
    loaded = persistence.load_camera_array(file_path)

    # Assert equivalence
    assert len(loaded.cameras) == len(original.cameras)
    for port in original.cameras:
        orig_cam = original.cameras[port]
        loaded_cam = loaded.cameras[port]

        assert loaded_cam.port == orig_cam.port
        assert loaded_cam.size == orig_cam.size
        assert loaded_cam.rotation_count == orig_cam.rotation_count
        assert loaded_cam.error == orig_cam.error
        assert loaded_cam.exposure == orig_cam.exposure
        assert loaded_cam.grid_count == orig_cam.grid_count
        assert loaded_cam.ignore == orig_cam.ignore
        assert loaded_cam.fisheye == orig_cam.fisheye

        # Check numpy arrays
        if orig_cam.matrix is not None:
            np.testing.assert_array_equal(loaded_cam.matrix, orig_cam.matrix)
        else:
            assert loaded_cam.matrix is None

        if orig_cam.distortions is not None:
            np.testing.assert_array_equal(loaded_cam.distortions, orig_cam.distortions)
        else:
            assert loaded_cam.distortions is None

        if orig_cam.translation is not None:
            np.testing.assert_array_equal(loaded_cam.translation, orig_cam.translation)
        else:
            assert loaded_cam.translation is None

        if orig_cam.rotation is not None:
            # floating point imprecision from rodrigues calcultion -> not exactly equal
            np.testing.assert_allclose(loaded_cam.rotation, orig_cam.rotation, rtol=1e-10, atol=1e-10)
        else:
            assert loaded_cam.rotation is None

    logger.info("✓ CameraArray round-trip test passed")


def test_charuco_roundtrip(tmp_path: Path):
    """Test Charuco save/load round-trip."""
    logger.info("Testing Charuco round-trip...")

    # Create test data
    original = create_test_charuco()
    file_path = tmp_path / "charuco.toml"

    # Save and load
    persistence.save_charuco(original, file_path)
    loaded = persistence.load_charuco(file_path)

    # Assert equivalence
    assert loaded.columns == original.columns
    assert loaded.rows == original.rows
    assert loaded.board_height == original.board_height
    assert loaded.board_width == original.board_width
    assert loaded.dictionary == original.dictionary
    assert loaded.units == original.units
    assert loaded.aruco_scale == original.aruco_scale
    assert loaded.square_size_overide_cm == original.square_size_overide_cm
    assert loaded.inverted == original.inverted
    assert loaded.legacy_pattern == original.legacy_pattern

    logger.info("✓ Charuco round-trip test passed")


def test_point_estimates_roundtrip(tmp_path: Path):
    """Test PointEstimates save/load round-trip using real fixture data."""
    logger.info("Testing PointEstimates round-trip...")

    # Load real fixture data
    fixtures = load_fixture_data()
    original = fixtures["point_estimates"]
    file_path = tmp_path / "point_estimates.toml"

    # Save and load
    persistence.save_point_estimates(original, file_path)
    loaded = persistence.load_point_estimates(file_path)

    # Assert equivalence
    np.testing.assert_array_equal(loaded.sync_indices, original.sync_indices)
    np.testing.assert_array_equal(loaded.camera_indices, original.camera_indices)
    np.testing.assert_array_equal(loaded.point_id, original.point_id)
    np.testing.assert_array_equal(loaded.img, original.img)
    np.testing.assert_array_equal(loaded.obj_indices, original.obj_indices)
    np.testing.assert_array_equal(loaded.obj, original.obj)

    logger.info("✓ PointEstimates round-trip test passed")


def test_paired_pose_network_roundtrip(tmp_path: Path):
    """Test PairedPoseNetwork save/load round-trip using real fixture data."""
    logger.info("Testing PairedPoseNetwork round-trip...")

    # Load real fixture data
    fixtures = load_fixture_data()
    original = fixtures["paired_pose_network"]
    file_path = tmp_path / "stereo_pairs.toml"

    # Save and load
    persistence.save_stereo_pairs(original, file_path)
    loaded = persistence.load_stereo_pairs(file_path)

    # Assert equivalence
    assert len(loaded._pairs) == len(original._pairs)

    for (port_a, port_b), original_pair in original._pairs.items():
        loaded_pair = loaded._pairs.get((port_a, port_b))
        assert loaded_pair is not None, f"Pair ({port_a}, {port_b}) missing after load"

        assert loaded_pair.primary_port == original_pair.primary_port
        assert loaded_pair.secondary_port == original_pair.secondary_port
        assert loaded_pair.error_score == original_pair.error_score

        np.testing.assert_allclose(loaded_pair.rotation, original_pair.rotation, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(loaded_pair.translation, original_pair.translation, rtol=1e-10, atol=1e-10)

    logger.info("✓ PairedPoseNetwork round-trip test passed")


def test_image_points_roundtrip(tmp_path: Path):
    """Test ImagePoints CSV save/load round-trip using real fixture data."""
    logger.info("Testing ImagePoints CSV round-trip...")

    # Load real fixture data
    fixtures = load_fixture_data()
    original = fixtures["image_points"]
    file_path = tmp_path / "xy_test.csv"

    # Save and load
    persistence.save_image_points_csv(original, file_path)
    loaded = persistence.load_image_points_csv(file_path)

    # Assert equivalence
    pd.testing.assert_frame_equal(loaded.df, original.df, check_dtype=True)

    logger.info("✓ ImagePoints CSV round-trip test passed")


if __name__ == "__main__":
    import tempfile

    # Setup logging for manual execution
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        logger.info("=" * 60)
        logger.info("RUNNING PERSISTENCE ROUND-TRIP TESTS")
        logger.info("=" * 60)

        try:
            test_camera_array_roundtrip(tmp_path)
            test_charuco_roundtrip(tmp_path)
            test_point_estimates_roundtrip(tmp_path)
            test_paired_pose_network_roundtrip(tmp_path)
            test_image_points_roundtrip(tmp_path)

            logger.info("=" * 60)
            logger.info("ALL TESTS PASSED")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"TEST FAILED: {e}")
            raise
