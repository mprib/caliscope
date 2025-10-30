import logging
from pathlib import Path

import numpy as np

from caliscope import __root__
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    create_point_estimates_from_stereopairs,
)
from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.cameras.camera_array import CameraArray
from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents

logger = logging.getLogger(__name__)


def test_point_estimates_structure_fully_linked():
    """
    Tests the structural integrity of PointEstimates from a fully-linked
    calibration where all cameras are successfully posed.
    """

    # Define source and a fresh destination for test data
    original_session_path = Path(__root__, "tests", "sessions", "point_estimate_creation", "fully_linked")
    session_path = Path(__root__, "tests", "sessions_copy_delete", "point_estimate_creation", "fully_linked")

    # Create a fresh copy of the data for the test
    copy_contents(original_session_path, session_path)

    config = Configurator(session_path)
    xy_data_path = Path(session_path, "xy_CHARUCO.csv")
    camera_array = config.get_camera_array()
    stereo_results = StereoCalibrator(camera_array, xy_data_path).stereo_calibrate_all()
    initializer = CameraArrayInitializer(camera_array, stereo_results)
    # This initialization should result in all cameras being posed
    camera_array: CameraArray = initializer.get_best_camera_array()

    # Generate the point estimates
    point_estimates = create_point_estimates_from_stereopairs(camera_array, xy_data_path)

    # --- Structural Integrity Assertions ---
    # 1. Check consistency between CameraArray and PointEstimates
    # The number of cameras in PointEstimates must match the number of posed cameras.
    n_posed_cameras = len(camera_array.posed_cameras)
    assert point_estimates.n_cameras == n_posed_cameras
    logger.info(f"OK: n_cameras matches posed cameras ({n_posed_cameras}).")

    # 2. Check 2D Observation Array Consistency
    # All arrays related to the 2D observations must have the same length.
    n_img_points = point_estimates.n_img_points
    assert point_estimates.img.shape[0] == n_img_points
    assert len(point_estimates.camera_indices) == n_img_points
    assert len(point_estimates.obj_indices) == n_img_points
    assert len(point_estimates.sync_indices) == n_img_points
    assert len(point_estimates.point_id) == n_img_points
    logger.info(f"OK: All 2D observation arrays have a consistent length of {n_img_points}.")

    # 3. Check 3D Point Array Consistency
    # The number of 3D points should be consistent.
    n_obj_points = point_estimates.n_obj_points
    assert point_estimates.obj.shape[0] == n_obj_points
    logger.info(f"OK: 3D point array has a consistent length of {n_obj_points}.")

    # 4. Check Index Range Validity
    # Camera indices must be zero-based and contiguous, from 0 to n_cameras-1.
    assert point_estimates.camera_indices.min() == 0
    assert point_estimates.camera_indices.max() == n_posed_cameras - 1
    logger.info(f"OK: Camera indices are in the expected range [0, {n_posed_cameras - 1}].")

    # Object indices must be zero-based and contiguous, from 0 to n_obj_points-1.
    assert point_estimates.obj_indices.min() == 0
    assert point_estimates.obj_indices.max() == n_obj_points - 1
    logger.info(f"OK: Object indices are in the expected range [0, {n_obj_points - 1}].")

    # 5. Check for "Dangling" 3D Points
    # Every 3D point in the `obj` array must be referenced at least once.
    # If this fails, it means there are unused 3D points in the `obj` array.
    unique_referenced_obj_indices = np.unique(point_estimates.obj_indices)
    assert len(unique_referenced_obj_indices) == n_obj_points
    logger.info("OK: All 3D points are referenced by at least one 2D observation.")


def test_point_estimates_structure_unlinked():
    """
    Tests the structural integrity of PointEstimates when one camera is
    unlinked.
    """

    # Define source and a fresh destination for test data
    original_session_path = Path(__root__, "tests", "sessions", "point_estimate_creation", "unlinked_camera")
    session_path = Path(__root__, "tests", "sessions_copy_delete", "point_estimate_creation", "unlinked_camera")

    # Create a fresh copy of the data for the test
    copy_contents(original_session_path, session_path)
    session_path = Path(__root__, "tests", "sessions", "point_estimate_creation", "unlinked_camera")

    config = Configurator(session_path)
    xy_data_path = Path(session_path, "xy_CHARUCO.csv")

    # This initialization will result in one camera being unposed
    camera_array = config.get_camera_array()
    stereo_results = StereoCalibrator(camera_array, xy_data_path).stereo_calibrate_all()
    initializer = CameraArrayInitializer(camera_array, stereo_results)
    # This initialization should result in all cameras being posed
    camera_array: CameraArray = initializer.get_best_camera_array()

    # Generate the point estimates
    point_estimates = create_point_estimates_from_stereopairs(camera_array, xy_data_path)

    # --- Structural Integrity Assertions ---
    # 1. Check consistency between CameraArray and PointEstimates
    # The number of cameras in PointEstimates must match the number of posed cameras.
    n_posed_cameras = len(camera_array.posed_cameras)
    assert point_estimates.n_cameras == n_posed_cameras
    logger.info(f"OK: n_cameras matches posed cameras ({n_posed_cameras}).")

    # 2. Check 2D Observation Array Consistency
    # All arrays related to the 2D observations must have the same length.
    n_img_points = point_estimates.n_img_points
    assert point_estimates.img.shape[0] == n_img_points
    assert len(point_estimates.camera_indices) == n_img_points
    assert len(point_estimates.obj_indices) == n_img_points
    assert len(point_estimates.sync_indices) == n_img_points
    assert len(point_estimates.point_id) == n_img_points
    logger.info(f"OK: All 2D observation arrays have a consistent length of {n_img_points}.")

    # 3. Check 3D Point Array Consistency
    # The number of 3D points should be consistent.
    n_obj_points = point_estimates.n_obj_points
    assert point_estimates.obj.shape[0] == n_obj_points
    logger.info(f"OK: 3D point array has a consistent length of {n_obj_points}.")

    # 4. Check Index Range Validity
    # Camera indices must be zero-based and contiguous, from 0 to n_cameras-1.
    assert point_estimates.camera_indices.min() == 0
    assert point_estimates.camera_indices.max() == n_posed_cameras - 1
    logger.info(f"OK: Camera indices are in the expected range [0, {n_posed_cameras - 1}].")

    # Object indices must be zero-based and contiguous, from 0 to n_obj_points-1.
    assert point_estimates.obj_indices.min() == 0
    assert point_estimates.obj_indices.max() == n_obj_points - 1
    logger.info(f"OK: Object indices are in the expected range [0, {n_obj_points - 1}].")

    # 5. Check for "Dangling" 3D Points
    # Every 3D point in the `obj` array must be referenced at least once.
    # If this fails, it means there are unused 3D points in the `obj` array.
    unique_referenced_obj_indices = np.unique(point_estimates.obj_indices)
    assert len(unique_referenced_obj_indices) == n_obj_points
    logger.info("OK: All 3D points are referenced by at least one 2D observation.")


if __name__ == "__main__":
    # test_point_estimates_structure_fully_linked()
    test_point_estimates_structure_unlinked()
    logger.info("End ad hoc test")
