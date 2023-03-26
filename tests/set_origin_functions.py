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



# Proceeding with basic idea that these functions will go into CaptureVolume.
# so use capture_volume here which may just become self later on.


def get_world_corners_xyz(capture_volume: CaptureVolume, sync_index: int) -> np.ndarray:
    """
    returns the estimated x,y,z position of the board corners at the given sync index
    note that the array is ordered according to the charuco id
    """
    sync_indices = capture_volume.point_estimates.sync_indices  # convienent shortening

    charuco_ids = capture_volume.point_estimates.point_id[sync_indices == sync_index]
    unique_charuco_id = np.unique(charuco_ids)
    unique_charuco_id.sort()

    # pull out the 3d point estimate indexes associated with the chosen sync_index
    # note that this will include duplicates
    obj_indices = capture_volume.point_estimates.obj_indices[sync_indices == sync_index]
    # now get the actual x,y,z estimate associated with these unique charucos
    obj_xyz = capture_volume.point_estimates.obj[obj_indices]

    sorter = np.argsort(charuco_ids)
    # need to get charuco ids associated with the 3 point positions
    unique_charuco_xyz_index = sorter[
        np.searchsorted(charuco_ids, unique_charuco_id, sorter=sorter)
    ]
    world_corners_xyz = obj_xyz[unique_charuco_xyz_index]
    return world_corners_xyz


def get_board_corners_xyz(
    capture_volume: CaptureVolume, sync_index: int, charuco: Charuco
) -> np.ndarray:
    """
    Returns corner positions in board world (x,y,0) for the corners with estimated point
    coordinates at the give sync_index
    note that the array is ordered according to the charuco id
    """
    sync_indices = capture_volume.point_estimates.sync_indices  # convienent shortening
    charuco_ids = capture_volume.point_estimates.point_id[sync_indices == sync_index]
    unique_charuco_id = np.unique(charuco_ids)
    unique_charuco_id.sort()

    board_corners_xyz = charuco.board.chessboardCorners[unique_charuco_id]
    return board_corners_xyz


def get_anchor_camera(capture_volume: CaptureVolume, sync_index: int) -> CameraData:
    """
    Returns the camera data object that viewed the most board corners
    """
    sync_indices = capture_volume.point_estimates.sync_indices  # convienent shortening
    camera_views = capture_volume.point_estimates.camera_indices[
        sync_indices == sync_index
    ]
    camera_port, camera_count = np.unique(camera_views, return_counts=True)
    anchor_camera_port = camera_port[camera_count.argmax()]

    anchor_camera: CameraData = camera_array.cameras[anchor_camera_port]
    return anchor_camera


def get_initial_origin_transform(capture_volume: CaptureVolume, sync_index: int, charuco:Charuco):

    world_corners_xyz = get_world_corners_xyz(capture_volume, sync_index)
    board_corners_xyz = get_board_corners_xyz(capture_volume, sync_index, charuco)
    anchor_camera = get_anchor_camera(capture_volume, sync_index)

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

    return origin_shift_transform


def shift_capture_volume_origin(
    capture_volume: CaptureVolume, origin_shift_transform: np.ndarray
) -> CaptureVolume:

    # update 3d point estimates
    xyz = capture_volume.point_estimates.obj
    scale = np.expand_dims(np.ones(xyz.shape[0]), 1)
    xyzh = np.hstack([xyz, scale])

    new_origin_xyzh = np.matmul(np.linalg.inv(origin_shift_transform), xyzh.T).T
    capture_volume.point_estimates.obj = new_origin_xyzh[:, 0:3]

    # update camera array
    for port, camera_data in capture_volume.camera_array.cameras.items():
        camera_data.transformation = np.matmul(
            camera_data.transformation, origin_shift_transform
        )
        
    return capture_volume



if __name__ == "__main__":

    # test_scenario = "4_cameras_nonoverlap"
    # test_scenario = "3_cameras_middle"
    test_scenario = "3_cameras_triangular"
    # test_scenario = "4_cameras_beginning" # initial translation off
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

    # need to get the charuco board that was used during the session for later
    session = Session(session_directory)
    charuco = session.charuco


    REOPTIMIZE_CAPTURE_VOLUME = True
    # REOPTIMIZE_CAPTURE_VOLUME = False

    if REOPTIMIZE_CAPTURE_VOLUME:

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


    origin_sync_index = origin_sync_indices[test_scenario]
    logger.warning(f"New test sync index is {origin_sync_index}")
    
    origin_transform = get_initial_origin_transform(capture_volume,origin_sync_index, charuco) 

    capture_volume = shift_capture_volume_origin(capture_volume,origin_transform)
    
    app = QApplication(sys.argv)
    vizr = CaptureVolumeVisualizer(capture_volume=capture_volume)
    # vizr = CaptureVolumeVisualizer(camera_array = capture_volume.camera_array)

    vizr_dialog = CaptureVolumeDialog(vizr)
    vizr_dialog.show()

    sys.exit(app.exec())