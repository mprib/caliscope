#%%
from pathlib import Path
from PyQt6.QtWidgets import QApplication
import sys

from pyxy3d.session import Session
from pyxy3d import __root__
from pyxy3d.calibration.stereocalibrator import StereoCalibrator
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)

from pyxy3d.gui.vizualize.capture_volume_visualizer import CaptureVolumeVisualizer

# the session used for the single camera calibration and
# the stereoframe data collection
# session_directory = Path(__root__,"tests", "3_cameras_linear" )


# NOTE: This will successfully calibrate when I fix my issue
# session_directory = Path(__root__,"tests", "3_cameras_middle" )
session_directory = Path(__root__, "tests", "4_cameras_beginning")
# session_directory = Path(__root__,"tests", "3_cameras_triangular" )
# session_directory = Path(__root__,"tests", "3_cameras_midlinear" )
# session_directory = Path(__root__,"tests", "2_cameras_linear" )
# session_directory = Path(__root__,"tests", "2_cameras_90_deg" )
# session_directory = Path(__root__,"tests", "2_cameras_gt_90" )
# session_directory = Path(__root__,"tests", "just_checking" )

# point_data.csv is created during the stereoframe datacollection
point_data_path = Path(session_directory, "point_data.csv")

# where single camera data is pulled from annd
# stereopair data is updated to
config_path = Path(session_directory, "config.toml")

# with the point data, the stereocalibrations can be performed.
# Note that this is named StereoCalibrator because it has a poorly
# working single camera calibration method as well, but I wouldn't recommend
# might be better termed stereocalibrator going forward

stereocalibrator = StereoCalibrator(config_path, point_data_path)

# create the pairwise estimates of camera positions
# this will save out "stereo_A_B" data to the config.toml
stereocalibrator.stereo_calibrate_all(boards_sampled=20)

# the pairwise estimates will be used to create the initial
# estimate of the camera array positions (now in the config file)

array_initializer = CameraArrayInitializer(config_path)
camera_array = array_initializer.get_best_camera_array()
point_estimates = get_point_estimates(camera_array, point_data_path)

capture_volume = CaptureVolume(camera_array, point_estimates)
capture_volume.save(session_directory)
capture_volume.optimize()
capture_volume.save(session_directory)
# The 3D point estimates are constructed from the estimated camera
# array. Stereopair triangulations are made and
# averaged together for each point.
# This is used to initialize the bundle adjustment
point_estimates: PointEstimates = get_point_estimates(camera_array, point_data_path)

# %%
# session.save_camera_array()
# capture_volume.save(session_directory)

app = QApplication(sys.argv)
vizr = CaptureVolumeVisualizer(capture_volume=capture_volume)
sys.exit(app.exec())
