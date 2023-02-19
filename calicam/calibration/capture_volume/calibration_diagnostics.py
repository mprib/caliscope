
import calicam.logger
logger = calicam.logger.get(__name__)

from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
import toml
from calicam.calibration.charuco import Charuco
from calicam.cameras.camera_array import CaptureVolume

CAMERA_PARAM_COUNT = 6



def get_charuco(config_path):
    config = toml.load(config_path)

    ## create charuco
    charuco = Charuco(
        columns=config["charuco"]["columns"],
        rows=config["charuco"]["rows"],
        board_height=config["charuco"]["board_height"],
        board_width=config["charuco"]["rows"],
        dictionary=config["charuco"]["dictionary"],
        units=config["charuco"]["units"],
        aruco_scale=config["charuco"]["aruco_scale"],
        square_size_overide_cm=config["charuco"]["square_size_overide_cm"],
        inverted=config["charuco"]["inverted"],
    )

    return charuco


def load_array_points_error(array_points_error_path):
    with open(array_points_error_path, "rb") as file:
        data = pickle.load(file)

    return data


def create_summary_df(diagnostic_data_path: Path, label):
    """
    Unpack the Array Diagnostic data into a pandas dataframe format that can be
    plotted and summarized. This is an omnibus dataframe that can be inspected for 
    accuracy and form the basis of additional slices used for a given purpose. 
    
    In particular, this data will get reformatted and paired down to create the inputs
    used for the Charuco corner distance calculation.
    """

    array_points_error_data = load_array_points_error(diagnostic_data_path)
    array_data_xy_error = array_points_error_data.xy_reprojection_error.reshape(-1, 2)
    # build out error as singular distanc

    xyz = array_points_error_data.get_xyz_points()

    euclidean_distance_error = np.sqrt(np.sum(array_data_xy_error**2, axis=1))
    row_count = euclidean_distance_error.shape[0]

    array_data_dict = {
        "label": [label] * row_count,
        "camera": array_points_error_data.point_estimate_data.camera_indices_full.tolist(),
        "sync_index": array_points_error_data.point_estimate_data.sync_indices.astype(int).tolist(),
        "charuco_id": array_points_error_data.point_estimate_data.corner_id.tolist(),
        "img_x": array_points_error_data.point_estimate_data.img_full[:, 0].tolist(),
        "img_y": array_points_error_data.point_estimate_data.img_full[:, 1].tolist(),
        "reproj_error_x": array_data_xy_error[:, 0].tolist(),
        "reproj_error_y": array_data_xy_error[:, 1].tolist(),
        "reproj_error": euclidean_distance_error.tolist(),
        "obj_id": array_points_error_data.point_estimate_data.obj_indices.tolist(),
        "obj_x": xyz[array_points_error_data.point_estimate_data.obj_indices_full][:, 0].tolist(),
        "obj_y": xyz[array_points_error_data.point_estimate_data.obj_indices_full][:, 1].tolist(),
        "obj_z": xyz[array_points_error_data.point_estimate_data.obj_indices_full][:, 2].tolist(),
    }

    summarized_data = (pd.DataFrame(array_data_dict)
                        .astype({"sync_index":'int32', "charuco_id":"int32", "obj_id":"int32"})
    )
    return summarized_data

def get_corners_xyz(config_path, diagnostic_data_path, label):
    all_session_data = create_summary_df(diagnostic_data_path, label)

    corners_3d = (all_session_data[
        ["label", "sync_index", "charuco_id", "obj_id", "obj_x", "obj_y", "obj_z"]
    ]
                    .groupby(["label","obj_id"])
                    .mean()
                    .reset_index()
                    .astype({"sync_index":'int32', "charuco_id":"int32", "obj_id":"int32"})
    )
    
    return corners_3d


### Moved to ArrayPointsErrorData dataclass
def get_xyz_points(capture_volume: CaptureVolume):
    """Get 3d positions arrived at by bundle adjustment"""
    n_cameras = len(capture_volume.camera_array.cameras)
    xyz = capture_volume.model_params[n_cameras * CAMERA_PARAM_COUNT :]
    xyz = xyz.reshape(-1, 3)

    return xyz


def get_xyz_ids(capture_volume: CaptureVolume):
    """get the charuco ids of the 3d points estimated by the bundle adjustment"""
    return capture_volume.point_estimate_data.obj_corner_id


if __name__ == "__main__":

    from calicam.cameras.camera_array import CaptureVolume
    from calicam import __root__
    # which enables import of relevant class

    calibration_directory = Path(__root__, "tests", "5_cameras", "recording")

    before_path = Path(calibration_directory, "before_bund_adj.pkl")
    after_path = Path(calibration_directory, "after_bund_adj.pkl")

    before_df = create_summary_df(before_path, "before")
    after_df = create_summary_df(after_path, "after")

    before_and_after = pd.concat([before_df, after_df])

    print(before_and_after.groupby(["label"])["reproj_error"].describe())

    config_path = Path(calibration_directory.parent, "config.toml")
    # charuco = get_charuco(config_path)

    # print(charuco.board.chessboardCorners)
