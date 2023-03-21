# I know that this is not really how tests are structured, but I'm just
# trying to begin getting in the habit of writing a test in a separate file
# from the code I'm developing...

# %%
from pathlib import Path
import numpy as np

from pyxy3d import __root__
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import \
    get_point_estimates
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates    
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.session import Session
from pyxy3d.calibration.charuco import Charuco
from cv2 import aruco

session_directory = Path(__root__, "tests", "4_cameras_endofday")

point_data_csv_path = Path(session_directory, "point_data.csv")

config_path = Path(session_directory, "config.toml")
array_initializer = CameraArrayInitializer(config_path)
camera_array = array_initializer.get_best_camera_array()
point_estimates = get_point_estimates(camera_array, point_data_csv_path)

print(f"Optimizing initial camera array configuration ")

capture_volume = CaptureVolume(camera_array, point_estimates)
capture_volume.save(session_directory)
capture_volume.optimize()
capture_volume.save(session_directory)

# %%

# config_path = Path(session_directory, "config.toml")
session = Session(session_directory)
charuco_board = session.charuco.board

# %%
sync_indices = point_estimates.sync_indices
test_sync_index = sync_indices[3]
charuco_ids = point_estimates.point_id
board_points = charuco_ids[sync_indices==test_sync_index]

obj_indices = point_estimates.obj_indices[sync_indices ==test_sync_index]
obj_xyz = point_estimates.obj[obj_indices]
# %%


# copied from https://longervision.github.io/2017/03/12/ComputerVision/OpenCV/opencv-external-posture-estimation-ArUco-board/
# retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs)  # posture estimation from a diamond
# Note for tomorrow: this is going to be more challenging than I'd thought..
# board pose is estimated from each camera...
# may need to get pose from each camera, then convert to a common
# frame of reference, and then average together....