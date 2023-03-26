
# %%
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from pathlib import Path
import numpy as np
import sys
import scipy
from PyQt6.QtWidgets import QApplication
from pyxy3d import __root__
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.session import Session
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.cameras.camera_array import CameraData
import cv2
from pyxy3d.gui.vizualize.capture_volume_visualizer import CaptureVolumeVisualizer
from pyxy3d.gui.vizualize.capture_volume_dialog import CaptureVolumeDialog
import pickle


test_scenario = "4_cameras_nonoverlap"
# test_scenario = "3_cameras_middle"
# test_scenario = "3_cameras_triangular"
# test_scenario = "4_cameras_beginning"
# test_scenario = "3_cameras_midlinear"


anchor_camera_override = None
# anchor_camera_override = 2

origin_sync_indices = {
    "4_cameras_nonoverlap": 23,
    "3_cameras_middle": 20,
    "4_cameras_beginning": 234,
    "3_cameras_triangular": 25,
    "3_cameras_midlinear": 14,
}

session_directory = Path(__root__, "tests", test_scenario)
point_data_csv_path = Path(session_directory, "point_data.csv")
config_path = Path(session_directory, "config.toml")

REOPTIMIZE_ARRAY = True
# REOPTIMIZE_ARRAY = False

if REOPTIMIZE_ARRAY:

    array_initializer = CameraArrayInitializer(config_path)
    camera_array = array_initializer.get_best_camera_array()
    point_estimates = get_point_estimates(camera_array, point_data_csv_path)

    print(f"Optimizing initial camera array configuration ")

    capture_volume = CaptureVolume(camera_array, point_estimates)
    capture_volume.save(session_directory, "initial")
    capture_volume.optimize()
    capture_volume.save(session_directory, "optimized")
else:

    saved_CV_path = Path(session_directory, "capture_volume_stage_1_optimized.pkl")
    with open(saved_CV_path, "rb") as f:
        capture_volume: CaptureVolume = pickle.load(f)
    point_estimates = capture_volume.point_estimates
    camera_array = capture_volume.camera_array

    
