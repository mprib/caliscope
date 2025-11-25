import matplotlib
import json

# Force non-interactive backend to prevent the debugger
# from trying to hook into the Qt GUI event loop.
matplotlib.use("Agg")


import logging
from pathlib import Path
from time import sleep


from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    create_point_estimates_from_stereopairs,
)
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.capture_volume.quality_controller import QualityController

from caliscope.calibration.array_initialization.legacy_stereocalibrator import LegacyStereoCalibrator
from caliscope.calibration.array_initialization.stereopair_graph import StereoPairGraph

# from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.configurator import Configurator
from caliscope.controller import FILTERED_FRACTION
from caliscope.helper import copy_contents
from caliscope.post_processing.point_data import ImagePoints
from caliscope.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.trackers.charuco_tracker import CharucoTracker


logger = logging.getLogger(__name__)


def test_xy_charuco_creation():
    original_session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration")

    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        "mediapipe_calibration",
    )

    copy_contents(original_session_path, session_path)

    # This test begins with a set of cameras with calibrated intrinsics
    config = Configurator(session_path)
    # config_path = str(Path(session_path, "config.toml"))
    logger.info(f"Getting charuco from config at {config.config_toml_path}")
    charuco = config.get_charuco()
    charuco_tracker = CharucoTracker(charuco)

    # create a synchronizer based off of these stream pools
    logger.info("Creating RecordedStreamPool")
    recording_path = Path(session_path, "calibration", "extrinsic")
    point_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")

    camera_array = config.get_camera_array()
    sync_stream_manager = SynchronizedStreamManager(
        recording_dir=recording_path, all_camera_data=camera_array.cameras, tracker=charuco_tracker
    )
    sync_stream_manager.process_streams(fps_target=100)

    # need to wait for points.csv file to populate
    while not point_data_path.exists():
        logger.info("Waiting for point_data.csv to populate...")
        sleep(1)

    assert point_data_path.exists()


def test_calibration():
    version = "larger_calibration_post_monocal"
    # version = "larger_calibration_post_bundle_adjustment"  # needed for test_stereocalibrate
    original_session_path = Path(__root__, "tests", "sessions", version)
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        version,
    )
    copy_contents(original_session_path, session_path)
    config = Configurator(session_path)
    recording_path = Path(session_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")
    camera_array = config.get_camera_array()
    charuco = config.get_charuco()

    logger.info("Creating stereocalibrator")
    image_points = ImagePoints.from_csv(xy_data_path)
    stereocalibrator = LegacyStereoCalibrator(camera_array, image_points)

    logger.info("Initiating stereocalibration")
    stereo_graph: StereoPairGraph = stereocalibrator.stereo_calibrate_all(boards_sampled=10)

    # save new_raw_stereograph
    # new_raw_stereograph = {}
    # for key, pair in stereo_graph._pairs.items():
    #     new_raw_stereograph[str(key)] = {"rotation": str(pair.rotation), "translation": str(pair.translation)}
    #
    # new_raw_stereograph_path = __root__ / "tests/reference/stereograph_gold_standard/new_raw_stereograph.json"
    # with open(new_raw_stereograph_path, "w") as f:
    #     json.dump(new_raw_stereograph, f, indent=4)

    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    stereo_graph.apply_to(camera_array)

    # save initial extrinsics
    new_initial_camera_array = {}
    for port, cam in camera_array.cameras.items():
        new_initial_camera_array[port] = {"rotation": str(cam.rotation), "translation": str(cam.translation)}

    new_initial_camera_array_path = __root__ / "tests/reference/stereograph_gold_standard/new_initial_camera_array.json"

    with open(new_initial_camera_array_path, "w") as f:
        json.dump(new_initial_camera_array, f, indent=4)

    logger.info("Loading point estimates")
    image_points = ImagePoints.from_csv(xy_data_path)
    point_estimates: PointEstimates = create_point_estimates_from_stereopairs(camera_array, image_points)

    config.save_point_estimates(point_estimates)
    config.save_camera_array(camera_array)

    capture_volume = CaptureVolume(camera_array, point_estimates)

    logger.info("=========== INITIAL CAMERA ARRAY ==============")
    for port, cam in capture_volume.camera_array.cameras.items():
        logger.info(f" Cam {port}: rotation - {cam.rotation}, translation = {cam.translation}")

    # Before filtering - log initial point counts
    logger.info("========== POINT COUNT DIAGNOSTICS ==========")
    logger.info("Initial point counts:")
    logger.info(f"  3D points (obj.shape[0]): {capture_volume.point_estimates.obj.shape[0]}")
    logger.info(f"  2D observations (img.shape[0]): {capture_volume.point_estimates.img.shape[0]}")
    logger.info(f"  Camera indices length: {len(capture_volume.point_estimates.camera_indices)}")
    logger.info(f"  Saving to path: {config.point_estimates_toml_path}")

    quality_controller = QualityController(capture_volume, charuco)

    # Verify initial state
    assert capture_volume.stage == 0
    rmse_initial = capture_volume.rmse
    assert rmse_initial is not None
    assert "overall" in rmse_initial
    assert all(str(port) in rmse_initial for port in capture_volume.camera_array.cameras.keys())

    # Log initial RMSE values
    logger.info(f"Initial RMSE before optimization: {rmse_initial['overall']:.4f} pixels")
    logger.info("Per-camera initial RMSE values:")
    for port in capture_volume.camera_array.cameras.keys():
        logger.info(f"  Camera {port}: {rmse_initial[str(port)]:.4f} pixels")

    # First optimization stage - bundle adjustment
    logger.info("Performing bundle adjustment")
    capture_volume.optimize()
    assert capture_volume.stage == 1

    # Log post-bundle adjustment RMSE and improvement
    rmse_post_bundle_adj = capture_volume.rmse
    improvement = rmse_initial["overall"] - rmse_post_bundle_adj["overall"]
    percent_improvement = (improvement / rmse_initial["overall"]) * 100
    logger.info(f"RMSE after bundle adjustment: {rmse_post_bundle_adj['overall']:.4f} pixels")
    logger.info(f"Improvement: {improvement:.4f} pixels ({percent_improvement:.2f}%)")
    assert rmse_post_bundle_adj["overall"] <= rmse_initial["overall"]

    # Second stage - filter out worse points
    logger.info(f"Filtering out worse fitting {FILTERED_FRACTION * 100:.1f}% of points")
    quality_controller.filter_point_estimates(FILTERED_FRACTION)

    # After filtering - log filtered point counts
    logger.info("Point counts AFTER filtering:")
    logger.info(f"  3D points (obj.shape[0]): {capture_volume.point_estimates.obj.shape[0]}")
    logger.info(f"  2D observations (img.shape[0]): {capture_volume.point_estimates.img.shape[0]}")
    logger.info(f"  Camera indices length: {len(capture_volume.point_estimates.camera_indices)}")

    # Log post-filtering RMSE (before re-optimization)
    rmse_post_filter = capture_volume.rmse
    logger.info(f"RMSE after filtering (before re-optimization): {rmse_post_filter['overall']:.4f} pixels")

    # Final stage - re-optimize with filtered data
    logger.info("Re-optimizing with filtered data set")
    capture_volume.optimize()

    # Log final RMSE and total improvement
    rmse_final = capture_volume.rmse
    total_improvement = rmse_initial["overall"] - rmse_final["overall"]
    total_percent = (total_improvement / rmse_initial["overall"]) * 100
    filter_improvement = rmse_post_bundle_adj["overall"] - rmse_final["overall"]
    filter_percent = (filter_improvement / rmse_post_bundle_adj["overall"]) * 100

    logger.info(f"Final RMSE after filtering and re-optimization: {rmse_final['overall']:.4f} pixels")
    logger.info(f"Improvement from filtering: {filter_improvement:.4f} pixels ({filter_percent:.2f}%)")
    logger.info(f"Total improvement: {total_improvement:.4f} pixels ({total_percent:.2f}%)")
    logger.info("Per-camera final RMSE values:")
    for port in capture_volume.camera_array.cameras.keys():
        initial = rmse_initial[str(port)]
        final = rmse_final[str(port)]
        cam_improvement = (initial - final) / initial * 100
        logger.info(f"  Camera {port}: {final:.4f} pixels (improved {cam_improvement:.2f}%)")

    config.save_point_estimates(capture_volume.point_estimates)
    config.save_camera_array(capture_volume.camera_array)


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()

    # print("start")
    test_calibration()
    # print("end")
    # import pytest
    # pytest.main([__file__])
