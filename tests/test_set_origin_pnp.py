# I know that this is not really how tests are structured, but I'm just
# trying to begin getting in the habit of writing a test in a separate file
# from the code I'm developing...

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
from pyxy3d.gui.vizualize.capture_volume_widget import CaptureVolumeWidget
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

# config_path = Path(session_directory, "config.toml")
session = Session(session_directory)
charuco_board = session.charuco.board

sync_indices = point_estimates.sync_indices
origin_sync_index = origin_sync_indices[test_scenario]

logger.warning(f"New test sync index is {origin_sync_index}")

charuco_ids = point_estimates.point_id[sync_indices == origin_sync_index]
unique_charuco_id = np.unique(charuco_ids)
unique_charuco_id.sort()

# pull out the 3d point estimate indexes associated with the chosen sync_index
# note that this will include duplicates
obj_indices = point_estimates.obj_indices[sync_indices == origin_sync_index]
# now get the actual x,y,z estimate associated with these unique charucos
obj_xyz = point_estimates.obj[obj_indices]
sorter = np.argsort(charuco_ids)
# need to get charuco ids associated with the 3 point positions
unique_charuco_xyz_index = sorter[
    np.searchsorted(charuco_ids, unique_charuco_id, sorter=sorter)
]

world_corners_xyz = obj_xyz[unique_charuco_xyz_index]
# need to get x,y,z estimates in board world...
board_corners_xyz = charuco_board.chessboardCorners[unique_charuco_id]

#%%
###################### FIND ANCHOR CAMERA
# anchor camera will be the one that has the most actual views of the charuco board.
if anchor_camera_override is None:
    camera_views = point_estimates.camera_indices[sync_indices == origin_sync_index]
    camera_port, camera_count = np.unique(camera_views, return_counts=True)
    anchor_camera_port = camera_port[camera_count.argmax()]
else:
    anchor_camera_port = anchor_camera_override
    
anchor_camera: CameraData = camera_array.cameras[anchor_camera_port]

#%%
# find pose of anchor camera relative to board
# 
charuco_image_points, jacobian = cv2.projectPoints(
    world_corners_xyz,
    rvec=anchor_camera.rotation,
    tvec=anchor_camera.translation,
    cameraMatrix=anchor_camera.matrix,
    distCoeffs=np.array(
        [0, 0, 0, 0, 0], dtype=np.float32
    ),  # because points are via bundle adj., no distortion
)

# use solvepnp to estimate the pose of the camera relative to the board
# this provides a good estimate of rotation, but not of translation
retval, rvec, tvec = cv2.solvePnP(
    board_corners_xyz,
    charuco_image_points,
    cameraMatrix=anchor_camera.matrix,
    distCoeffs=np.array([0, 0, 0, 0, 0], dtype=np.float32),
)

rvec = cv2.Rodrigues(rvec)[0]

anchor_board_transform = np.hstack([rvec, tvec])
anchor_board_transform = np.vstack(
    [anchor_board_transform, np.array([0, 0, 0, 1], np.float32)]
)

#%%
# calculate the transformation matrix that will convert the anchor camera
# to the new frame of reference
origin_shift_transform = np.matmul(
    np.linalg.inv(anchor_camera.transformation), anchor_board_transform
)
# origin_shift_transform = np.matmul(anchor_board_transform, np.linalg.inv(anchor_camera.transformation))

# check application of origin_shift_transformation
#
check_transform = np.matmul(anchor_camera.transformation, origin_shift_transform)

logger.info(f"ChecK: Old method yielded transform of  \n {anchor_board_transform}")
logger.info(f"ChecK: New method yields transform of  \n {check_transform}")
#%%


#%%
######  SET CAMERA TRANSFORMATIONS
for port, camera_data in camera_array.cameras.items():

    charuco_image_points, jacobian = cv2.projectPoints(
        world_corners_xyz,
        rvec=camera_data.rotation,
        tvec=camera_data.translation,
        cameraMatrix=camera_data.matrix,
        distCoeffs=np.array(
            [0, 0, 0, 0, 0], dtype=np.float32
        ),  # because points are via bundle adj., no distortion
    )

    # use solvepnp and not estimate poseboard.....
    retval, rvec, tvec = cv2.solvePnP(
        board_corners_xyz,
        charuco_image_points,
        cameraMatrix=camera_data.matrix,
        distCoeffs=np.array([0, 0, 0, 0, 0], dtype=np.float32),
    )
    # convert rvec to 3x3 rotation matrix
    logger.info(f"Rotation vector is {rvec}")
    rvec = cv2.Rodrigues(rvec)[0]
    logger.info(f"Rotation vector is {rvec}")

    new_origin_transform = np.hstack([rvec, tvec])
    new_origin_transform = np.vstack(
        [new_origin_transform, np.array([0, 0, 0, 1], np.float32)]
    )

    logger.info(f"About to attempt to change camera at port {port}")
    # camera_data.transformation = new_origin_transform
    logger.info(
        f"Previous method (that works) has transform of \n {new_origin_transform}"
    )

    camera_data.transformation = np.matmul(
        camera_data.transformation, origin_shift_transform
    )

    logger.info(
        f"New method (that doesn't quite work) has transform of \n {camera_data.transformation}"
    )
##########################################################################
#%%


# test_new_origin_world_corners_xyz  =

# change the point estimates to reflect the new origin
xyz = capture_volume.point_estimates.obj
scale = np.expand_dims(np.ones(xyz.shape[0]), 1)
xyzh = np.hstack([xyz, scale])

new_origin_xyzh = np.matmul(np.linalg.inv(origin_shift_transform), xyzh.T).T
# new_origin_xyzh = np.matmul(board_pose_transformation,xyzh.T).T
capture_volume.point_estimates.obj = new_origin_xyzh[:, 0:3]


############## REASSESS POINT ESTIMATE ORIGIN FRAME ##################################

obj_indices = capture_volume.point_estimates.obj_indices[
    sync_indices == origin_sync_index
]
# now get the actual x,y,z estimate associated with these unique charucos
obj_xyz = capture_volume.point_estimates.obj[obj_indices]
sorter = np.argsort(charuco_ids)
# need to get charuco ids associated with the 3 point positions
unique_charuco_xyz_index = sorter[
    np.searchsorted(charuco_ids, unique_charuco_id, sorter=sorter)
]

new_cap_vol_world_corners_xyz = obj_xyz[unique_charuco_xyz_index]
# need to get x,y,z estimates in board world...
board_corners_xyz = charuco_board.chessboardCorners[unique_charuco_id]

capture_volume.save(session_directory, "new_origin")

logger.info("About to visualize the camera array")

# Here is the plan: from a given sync_index, find which camera has the most points represented on it.
# or wait...does this matter...can I just project back to the camera from the 3d points

#%%
## the problem here may have been rerunning the bundle adjustment from the initial points...
## it's just going to cause drift in the camera array...I need a better way to actually triangulate the data.
## as a check of the correctness of the transformation


# camera_array = capture_volume.camera_array
# point_estimates = get_point_estimates(camera_array, point_data_csv_path)
# capture_volume = CaptureVolume(camera_array, point_estimates)
# capture_volume.optimize()

app = QApplication(sys.argv)
vizr = CaptureVolumeVisualizer(capture_volume=capture_volume)
# vizr = CaptureVolumeVisualizer(camera_array = capture_volume.camera_array)

vizr_dialog = CaptureVolumeWidget(vizr)
vizr_dialog.show()

sys.exit(app.exec())
# %%
