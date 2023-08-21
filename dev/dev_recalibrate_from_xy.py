
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from time import sleep
import shutil
from pathlib import Path
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d import __root__
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
import pytest
from pyxy3d.calibration.charuco import Charuco, get_charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.calibration.monocalibrator import MonoCalibrator
from pyxy3d.cameras.camera import Camera
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer


from pyxy3d.calibration.stereocalibrator import StereoCalibrator
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.quality_controller import QualityController

from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)

from pyxy3d.cameras.live_stream import LiveStream
from pyxy3d.recording.video_recorder import VideoRecorder
from pyxy3d.recording.recorded_stream import RecordedStream, RecordedStreamPool

from pyxy3d.session.session import FILTERED_FRACTION
from pyxy3d.configurator import Configurator

target_session_path = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\20230818_do")
point_data_path = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\20230818_do\calibration\extrinsic\xy.csv")

config = Configurator(target_session_path)
logger.info(f"Waiting for video recorder to finish processing stream...")
stereocalibrator = StereoCalibrator(config.config_toml_path, point_data_path)
stereocalibrator.stereo_calibrate_all(boards_sampled=10)

camera_array: CameraArray = CameraArrayInitializer(
    config.config_toml_path
).get_best_camera_array()

point_estimates: PointEstimates = get_point_estimates(camera_array, point_data_path)

capture_volume = CaptureVolume(camera_array, point_estimates)
initial_rmse = capture_volume.rmse
logger.info(f"Prior to bundle adjustment, RMSE error is {initial_rmse}")
capture_volume.optimize()

charuco = config.get_charuco()
quality_controller = QualityController(capture_volume, charuco)
# Removing the worst fitting {FILTERED_FRACTION*100} percent of points from the model
logger.info(f"Filtering out worse fitting {FILTERED_FRACTION*100} % of points")
quality_controller.filter_point_estimates(FILTERED_FRACTION)
logger.info("Re-optimizing with filtered data set")
capture_volume.optimize()
optimized_filtered_rmse = capture_volume.rmse

# save out results of optimization for later assessment with F5 test walkthroughs
config.save_camera_array(capture_volume.camera_array)
config.save_point_estimates(capture_volume.point_estimates)

for key, optimized_rmse in optimized_filtered_rmse.items():
    logger.info(f"Asserting that RMSE decreased with optimization at {key}...")
    assert initial_rmse[key] > optimized_rmse