# %%
import caliscope.logger

logger = caliscope.logger.get(__name__)

from pathlib import Path
import numpy as np
import sys
import scipy
from PySide6.QtWidgets import QApplication
from caliscope import __root__
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.charuco import Charuco
from caliscope.cameras.camera_array import CameraData, CameraArray
import cv2
import pickle


# Proceeding with basic idea that these functions will go into CaptureVolume.
# so use capture_volume here which may just become self later on.


def get_world_corners_xyz(
    point_estimates: PointEstimates, sync_index: int
) -> np.ndarray:
    """
    returns the estimated x,y,z position of the board corners at the given sync index
    note that the array is ordered according to the charuco id
    """
    sync_indices = point_estimates.sync_indices  # convienent shortening

    charuco_ids = point_estimates.point_id[sync_indices == sync_index]
    unique_charuco_id = np.unique(charuco_ids)
    unique_charuco_id.sort()

    # pull out the 3d point estimate indexes associated with the chosen sync_index
    # note that this will include duplicates
    obj_indices = point_estimates.obj_indices[sync_indices == sync_index]
    # now get the actual x,y,z estimate associated with these unique charucos
    obj_xyz = point_estimates.obj[obj_indices]

    sorter = np.argsort(charuco_ids)
    # need to get charuco ids associated with the 3 point positions
    unique_charuco_xyz_index = sorter[
        np.searchsorted(charuco_ids, unique_charuco_id, sorter=sorter)
    ]
    world_corners_xyz = obj_xyz[unique_charuco_xyz_index]
    return world_corners_xyz


def get_board_corners_xyz(
    point_estimates: PointEstimates, sync_index: int, charuco: Charuco
) -> np.ndarray:
    """
    Returns corner positions in board world (x,y,0) for the corners with estimated point
    coordinates at the give sync_index
    note that the array is ordered according to the charuco id
    """
    sync_indices = point_estimates.sync_indices  # convienent shortening
    charuco_ids = point_estimates.point_id[sync_indices == sync_index]
    unique_charuco_id = np.unique(charuco_ids)
    unique_charuco_id.sort()

    board_corners_xyz = charuco.board.getChessboardCorners()[unique_charuco_id]
    return board_corners_xyz


def get_anchor_cameras(
    camera_array: CameraArray, point_estimates: PointEstimates, sync_index: int
) -> list:
    """
    Returns the camera data objects that have an actual view of the board
    at the sync index and therefore can be used to estimate the pose from pnp

    Weird results will happen if the camera has its back turned to the board and
    the corners are projected into it.

    """
    sync_indices = point_estimates.sync_indices  # convienent shortening
    camera_views = point_estimates.camera_indices[sync_indices == sync_index]
    camera_ports, camera_counts = np.unique(camera_views, return_counts=True)

    anchor_cameras = []
    for port in camera_ports:
        cam: CameraData = camera_array.cameras[port]
        anchor_cameras.append(cam)

    return anchor_cameras


def get_rvec_tvec_from_board_pose(
    camera_array: CameraArray,
    point_estimates: PointEstimates,
    sync_index: int,
    charuco: Charuco,
):
    world_corners_xyz = get_world_corners_xyz(point_estimates, sync_index)
    board_corners_xyz = get_board_corners_xyz(point_estimates, sync_index, charuco)
    anchor_cameras = get_anchor_cameras(camera_array, point_estimates, sync_index)

    rvecs = []
    tvecs = []

    for camera in anchor_cameras:
        charuco_image_points, jacobian = cv2.projectPoints(
            world_corners_xyz,
            rvec=camera.rotation,
            tvec=camera.translation,
            cameraMatrix=camera.matrix,
            distCoeffs=np.array(
                [0, 0, 0, 0, 0], dtype=np.float32
            ),  # because points are via bundle adj., no distortion
        )

        # use solvepnp to estimate the pose of the camera relative to the board
        # this provides a good estimate of rotation, but not of translation
        retval, rvec, tvec = cv2.solvePnP(
            board_corners_xyz,
            charuco_image_points,
            cameraMatrix=camera.matrix,
            distCoeffs=np.array([0, 0, 0, 0, 0], dtype=np.float32),
        )

        anchor_board_transform = rvec_tvec_to_transform(rvec, tvec)

        # back into the shift in the world change of origin implied by the
        # pose of the camera relative to the board given its previous
        # pose in the old frame of reference
        origin_shift_transform = np.matmul(
            np.linalg.inv(camera.transformation), anchor_board_transform
        )

        rvec, tvec = transform_to_rvec_tvec(origin_shift_transform)

        rvecs.append(rvec)
        tvecs.append(tvec)

    mean_rvec = mean_vec(rvecs)
    mean_tvec = mean_vec(tvecs)

    # get mean values
    return mean_rvec, mean_tvec


def mean_vec(vecs):
    hstacked_vec = None

    for vec in vecs:
        if hstacked_vec is None:
            hstacked_vec = vec
        else:
            hstacked_vec = np.hstack([hstacked_vec, vec])

    mean_vec = np.mean(hstacked_vec, axis=1)
    mean_vec = np.expand_dims(mean_vec, axis=1)

    return mean_vec


def transform_to_rvec_tvec(transformation: np.ndarray):
    rot_matrix = transformation[0:3, 0:3]
    rvec = cv2.Rodrigues(rot_matrix)[0]
    tvec = np.expand_dims(transformation[0:3, 3], axis=1)
    return rvec, tvec


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # might send a rotation matrix into here so check
    # that rvec is a rodrigues vector before converting
    if len(rvec.shape) == 1 or rvec.shape[1] == 1:
        rvec = cv2.Rodrigues(rvec)[0]

    # if tvec doesn't have the extra dimension, add it
    if len(tvec.shape) == 1:
        tvec = np.expand_dims(tvec, axis=1)

    transform = np.hstack([rvec, tvec])
    transform = np.vstack([transform, np.array([0, 0, 0, 1], np.float32)])
    return transform


def world_board_distance(
    tvec_xyz: np.ndarray, good_rvec: np.ndarray, raw_world_xyz, board_corners_xyz
):
    scale = np.expand_dims(np.ones(raw_world_xyz.shape[0]), 1)
    raw_world_xyzh = np.hstack([raw_world_xyz, scale])

    origin_shift_transform = rvec_tvec_to_transform(good_rvec, tvec_xyz)

    new_origin_xyzh = np.matmul(
        np.linalg.inv(origin_shift_transform), raw_world_xyzh.T
    ).T
    new_origin_xyz = new_origin_xyzh[:, 0:3]

    delta_xyz = new_origin_xyz - board_corners_xyz
    logger.info(f"Delta_xyz is {delta_xyz}")
    return delta_xyz.ravel()


def get_board_origin_transform(
    camera_array: CameraArray,
    point_estimates: PointEstimates,
    sync_index: int,
    charuco: Charuco,
):
    """
    Returns the 4x4 transformation matrix that will shift the capture volumes camera and
    point data to be represented with an origin that aligns with the origin of the charuco board.

    This is the primary function that is exposed from this file to the CaptureVolume class
    """
    # get initial approximation of the transformation to apply
    good_rvec, poor_tvec = get_rvec_tvec_from_board_pose(
        camera_array=camera_array,
        point_estimates=point_estimates,
        sync_index=sync_index,
        charuco=charuco,
    )

    world_board = get_world_corners_xyz(point_estimates, sync_index)
    target_board = get_board_corners_xyz(point_estimates, sync_index, charuco)
    initial_tvec = poor_tvec[:, 0]  # prep for least_squares array

    least_sq_result = scipy.optimize.least_squares(
        world_board_distance, initial_tvec, args=(good_rvec, world_board, target_board)
    )

    optimal_tvec = least_sq_result.x
    final_transform = rvec_tvec_to_transform(good_rvec, optimal_tvec)

    return final_transform


if __name__ == "__main__":
    #
    from caliscope.session.session import LiveSession
    from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
    from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
    from caliscope.gui.vizualize.calibration.capture_volume_visualizer import (
        CaptureVolumeVisualizer,
    )
    from caliscope.gui.vizualize.calibration.capture_volume_widget import (
        CaptureVolumeWidget,
    )

    # test_scenario = "4_cameras_nonoverlap"
    # test_scenario = "3_cameras_middle"
    # test_scenario = "2_cameras_linear"
    # test_scenario = "3_cameras_triangular"
    test_scenario = "4_cameras_beginning"  # initial translation off
    # test_scenario = "3_cameras_midlinear"

    anchor_camera_override = None
    # anchor_camera_override = 2

    origin_sync_indices = {
        "4_cameras_nonoverlap": 23,
        "2_cameras_linear": 77,
        "3_cameras_middle": 20,
        "4_cameras_beginning": 230,
        "3_cameras_triangular": 25,
        "3_cameras_midlinear": 14,
    }

    session_directory = Path(__root__, "tests", test_scenario)
    point_data_csv_path = Path(session_directory, "point_data.csv")
    config_path = Path(session_directory, "config.toml")

    # need to get the charuco board that was used during the session for later
    session = LiveSession(session_directory)
    tracker = session.charuco

    REOPTIMIZE_CAPTURE_VOLUME = True
    # REOPTIMIZE_CAPTURE_VOLUME = False

    if REOPTIMIZE_CAPTURE_VOLUME:
        array_initializer = CameraArrayInitializer(config_path)
        camera_array = array_initializer.get_best_camera_array()
        point_estimates = get_point_estimates(camera_array, point_data_csv_path)

        print(f"Optimizing initial camera array configuration ")

        capture_volume = CaptureVolume(camera_array, point_estimates)
        capture_volume._save(session_directory, "initial")
        capture_volume.optimize()
        capture_volume._save(session_directory, "optimized")
    else:
        saved_CV_path = Path(session_directory, "capture_volume_stage_1_optimized.pkl")
        with open(saved_CV_path, "rb") as f:
            capture_volume: CaptureVolume = pickle.load(f)

    origin_sync_index = origin_sync_indices[test_scenario]
    logger.warning(f"New test sync index is {origin_sync_index}")

    capture_volume.set_origin_to_board(origin_sync_index, tracker)

    app = QApplication(sys.argv)
    vizr = CaptureVolumeVisualizer(capture_volume=capture_volume)

    vizr_dialog = CaptureVolumeWidget(vizr)
    vizr_dialog.show()

    sys.exit(app.exec())
