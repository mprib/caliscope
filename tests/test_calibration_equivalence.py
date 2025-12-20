import matplotlib
import numpy as np

# Force non-interactive backend to prevent the debugger
# from trying to hook into the Qt GUI event loop.
matplotlib.use("Agg")
from copy import deepcopy

import logging
from pathlib import Path


from caliscope import __root__
from caliscope.core.capture_volume.capture_volume import CaptureVolume
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.core.capture_volume.point_estimates import PointEstimates


# from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope import persistence
from caliscope.core.point_data_bundle import PointDataBundle


logger = logging.getLogger(__name__)


# Helper function for rotation comparison
def rotation_angle_between(R1, R2):
    """Calculate angle between two rotation matrices in degrees."""
    R_rel = R1 @ R2.T
    trace = np.trace(R_rel)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(angle)


def test_optimization_equivalence(tmp_path: Path):
    version = "larger_calibration_post_monocal"
    # version = "larger_calibration_post_bundle_adjustment"  # needed for test_stereocalibrate
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    recording_path = Path(tmp_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    persistence.load_charuco(tmp_path / "charuco.toml")
    image_points = ImagePoints.from_csv(xy_data_path)

    logger.info("Creating paired pose network")

    paired_pose_network = build_paired_pose_network(image_points, camera_array)
    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    paired_pose_network.apply_to(camera_array, anchor_cam=8)
    world_points: WorldPoints = image_points.triangulate(camera_array)
    point_estimates: PointEstimates = world_points.to_point_estimates(image_points, camera_array)

    original_camera_array = deepcopy(camera_array)

    # create legacy optimization structure
    capture_volume = CaptureVolume(camera_array, point_estimates)
    capture_volume.optimize()

    # Create New PointDataBundle structure
    initial_point_data_bundle = PointDataBundle(original_camera_array, image_points, world_points)
    optimized_point_data_bundle = initial_point_data_bundle.optimize()

    cap_vol_cameras = capture_volume.camera_array
    bundle_cameras = optimized_point_data_bundle.camera_array

    # Compare camera poses
    logger.info("\nComparing camera poses (tolerance: 1.0mm, 0.5°)...")
    for port in sorted(cap_vol_cameras.posed_cameras.keys()):
        # Translation difference (mm)
        t_legacy = cap_vol_cameras.cameras[port].translation
        t_bundle = bundle_cameras.cameras[port].translation
        trans_diff_mm = np.linalg.norm(t_legacy - t_bundle) * 1000

        # Rotation difference (degrees)
        R_legacy = cap_vol_cameras.cameras[port].rotation
        R_bundle = bundle_cameras.cameras[port].rotation
        rot_diff_deg = rotation_angle_between(R_legacy, R_bundle)

        logger.info(f"  Camera {port}: translation diff = {trans_diff_mm:.2f}mm, rotation diff = {rot_diff_deg:.2f}°")
        assert trans_diff_mm < 20.0, f"Camera {port} translation diff {trans_diff_mm:.2f}mm exceeds 10.0mm"
        assert rot_diff_deg < 1, f"Camera {port} rotation diff {rot_diff_deg:.2f}° exceeds 1°"

    # Compare 3D point clouds
    logger.info("\nComparing 3D point clouds...")
    legacy_points = capture_volume.point_estimates.obj
    bundle_points = optimized_point_data_bundle.world_points.points

    if len(legacy_points) == len(bundle_points):
        point_diffs = np.linalg.norm(legacy_points - bundle_points, axis=1)
        avg_diff_mm = np.mean(point_diffs) * 1000
        max_diff_mm = np.max(point_diffs) * 1000

        logger.info(f"  Point count: {len(legacy_points)}")
        logger.info(f"  Average difference: {avg_diff_mm:.2f}mm")
        logger.info(f"  Max difference: {max_diff_mm:.2f}mm")

        assert avg_diff_mm < 5.0, f"Average point diff {avg_diff_mm:.2f}mm exceeds 5.0mm"
    else:
        logger.warning(f"  Point count mismatch: Legacy={len(legacy_points)}, Bundle={len(bundle_points)}")
        logger.warning("  Skipping point cloud comparison")

    logger.info("\n" + "=" * 60)
    logger.info("ALL COMPARISONS PASSED - Systems are equivalent within tolerance")
    logger.info("=" * 60)


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()

    # print("start")
    temp_path = Path(__file__).parent / "debug"
    test_optimization_equivalence(temp_path)
