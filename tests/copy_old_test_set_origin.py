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
from pyxy3d.gui.vizualize.capture_volume_dialog import CaptureVolumeDialog
import pickle

# session_directory = Path(__root__, "tests", "3_cameras_middle")
session_directory = Path(__root__, "tests", "4_cameras_beginning")
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
test_sync_index = sync_indices[46]
# test_sync_index = 35

charuco_ids = point_estimates.point_id[sync_indices == test_sync_index]
unique_charuco_id = np.unique(charuco_ids)
unique_charuco_id.sort()

# pull out the 3d point estimate indexes associated with the chosen sync_index
# note that this will include duplicates
obj_indices = point_estimates.obj_indices[sync_indices == test_sync_index]
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
# quick check of corner distances in world and board frame to make sure I'm not
# completely off track....
test_index_A = 0
test_index_B = 1

# get the distance between them
distance_world_A_B = np.sqrt(
    np.sum(
        (world_corners_xyz[test_index_A, :] - world_corners_xyz[test_index_B, :]) ** 2
    )
)

distance_board_A_B = np.sqrt(
    np.sum(
        (board_corners_xyz[test_index_A, :] - board_corners_xyz[test_index_B, :]) ** 2
    )
)

world_board_ratio = distance_world_A_B / distance_board_A_B
# adjust board_corners_xyz to reflect the scale of the world
board_corners_xyz = world_board_ratio * board_corners_xyz

#%%
# if True:


def board_distance_error(six_dof_params, board_corners_xyz, world_corners_xyz):
    """
    error function for estimating the transformation that will set the world origin
    to a board frame of reference.

    returns a vector of distances between each corner

    """
    rvec = cv2.Rodrigues(
        np.expand_dims(np.array([six_dof_params[0:3]], dtype=np.float32), 1)
    )[0]
    tvec = np.array([six_dof_params[3:]]).T

    new_origin_transform = np.hstack([rvec, tvec])
    new_origin_transform = np.vstack(
        [new_origin_transform, np.array([0, 0, 0, 1], np.float32)]
    )

    xyz = world_corners_xyz
    scale = np.expand_dims(np.ones(xyz.shape[0]), 1)
    xyzh = np.hstack([xyz, scale])

    new_origin_world_xyzh = np.matmul(np.linalg.inv(new_origin_transform), xyzh.T).T
    new_world_corners_xyz = new_origin_world_xyzh[:, 0:3]

    delta_xyz = board_corners_xyz - new_world_corners_xyz
    delta_xyz[2:, 0:2] = 0  # pin down 2 points for x,y control, otherwise ignore
    # delta_xyz[:,2] = abs(delta_xyz[:,2]) # make the algo care more about flatness

    minimize_target = delta_xyz.ravel()

    # distance_error = np.sqrt(np.sum((board_corners_xyz - new_world_corners_xyz)**2, axis=1))
    # if basin_hopping:
    # distance_error = np.sum(distance_error)
    # alternate approach here...just trying to drive the z coordinates to zero...
    # distance_error = new_world_corners_xyz[:,2]**2

    # logger.info(f"Distance Error during optimization: {distance_error}")
    # logger.info(f"Estimated Solution: {six_dof_params}")
    return minimize_target


six_dof_params_initial = [0, 0, 0, 0, 0, 0]
pi = 3.14159

bounds = ([0, 0, 0, -10, -10, -10], [pi, pi, pi, 10, 10, 10])

least_sq_result = scipy.optimize.least_squares(
    fun=board_distance_error,
    x0=six_dof_params_initial,
    # bounds=bounds,
    ftol=1e-10,
    args=[board_corners_xyz, world_corners_xyz],
)

six_dof_params = least_sq_result.x

rvec = cv2.Rodrigues(
    np.expand_dims(np.array(six_dof_params[0:3], dtype=np.float32), 1)
)[0]
# note that these translations result in the system moving in the negative direction
tvec = np.array([six_dof_params[3:]]).T


############################## POSSIBLE SOLUTION ON PAUSE ######################
# Commenting out code associated with attempts to use board pose....
# attempting alternate approach of calculating R|T that minimizes the difference
# between

# anchor_camera: CameraData = camera_array.cameras[list(camera_array.cameras.keys())[0]]

# charuco_image_points, jacobian = cv2.projectPoints(
#     world_corners_xyz,
#     rvec=anchor_camera.rotation,
#     tvec=anchor_camera.translation,
#     cameraMatrix=anchor_camera.matrix,
#     distCoeffs=np.array(
#         [0, 0, 0, 0, 0], dtype=np.float32
#     ),  # For origin setting, assume perfection
# )

# # use solvepnp and not estimate poseboard.....
# retval, rvec, tvec = cv2.solvePnP(
#     board_corners_xyz,
#     charuco_image_points,
#     cameraMatrix=anchor_camera.matrix,
#     distCoeffs=np.array([0, 0, 0, 0, 0], dtype=np.float32),
# )
# # convert rvec to 3x3 rotation matrix
# # logger.info(f"Rotation vector is {rvec}")
# rvec = cv2.Rodrigues(rvec)[0]
# # logger.info(f"Rotation vector is {rvec}")
# ##########################################################################


# I believe this is the transformation to be applied
# or perhaps the inverse, let's find out...
new_origin_transform = np.hstack([rvec, tvec])
new_origin_transform = np.vstack(
    [new_origin_transform, np.array([0, 0, 0, 1], np.float32)]
)

logger.info("About to attempt to change camera array")

for port, camera_data in camera_array.cameras.items():
    # camera_data.translation = camera_data.translation + tvec[:,0]
    # logger.info(f"Attempting to update camera at port {port}")
    old_transformation = camera_data.transformation
    new_transformation = np.dot(old_transformation, new_origin_transform)
    camera_data.transformation = new_transformation

old_world_corners_xyzh = np.hstack(
    [world_corners_xyz, np.expand_dims(np.ones(world_corners_xyz.shape[0]), 1)]
)
test_new_origin_world_corners_xyzh = np.matmul(
    np.linalg.inv(new_origin_transform), old_world_corners_xyzh.T
).T


# test_new_origin_world_corners_xyz  =

# change the point estimates to reflect the new origin
xyz = capture_volume.point_estimates.obj
scale = np.expand_dims(np.ones(xyz.shape[0]), 1)
xyzh = np.hstack([xyz, scale])

new_origin_xyzh = np.matmul(np.linalg.inv(new_origin_transform), xyzh.T).T
# new_origin_xyzh = np.matmul(board_pose_transformation,xyzh.T).T
capture_volume.point_estimates.obj = new_origin_xyzh[:, 0:3]


############## REASSESS POINT ESTIMATE ORIGIN FRAME ##################################

obj_indices = capture_volume.point_estimates.obj_indices[
    sync_indices == test_sync_index
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

camera_array = capture_volume.camera_array
point_estimates = get_point_estimates(camera_array, point_data_csv_path)
capture_volume = CaptureVolume(camera_array, point_estimates)

app = QApplication(sys.argv)
vizr = CaptureVolumeVisualizer(capture_volume=capture_volume)
# vizr = CaptureVolumeVisualizer(camera_array = capture_volume.camera_array)

vizr_dialog = CaptureVolumeDialog(vizr)
vizr_dialog.show()

sys.exit(app.exec())
# %%
