# import saved point data and initial array configuration data
# currently this is for the convenience of not having to rerun everything
# though this workflow may be useful into the future. Save out milestone calculations
# along the way that allow for blocks of dataprocessing 

from pathlib import Path

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PointData:

    camera_indices: np.ndarray
    img: np.ndarray
    obj_indices: np.ndarray
    obj: np.ndarray

    def filter(self, include):
        """Provide a vector of booleans...only keep the data associaed with True"""
        self.camera_indices = self.camera_indices[include]
        self.obj_indices = self.obj_indices[include]
        self.img = self.img[include]

        # if filtering means that some 3d points are no longer referenced, then remove 
        used_obj = np.unique(self.obj_indices)
        used_obj.sort()
        self.obj = self.obj[used_obj]

def get_camera_params(camera_array):
    """for each camera build the CAMERA_PARAM_COUNT element parameter index
    camera_params with shape (n_cameras, CAMERA_PARAM_COUNT)
    contains initial estimates of parameters for all cameras.
    First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
    next 3 components form a translation vector
    """
    camera_params = None
    for port, cam in camera_array.cameras.items():
        port_param = cam.to_vector()
        if camera_params is None:
            camera_params = port_param
        else:
            camera_params = np.vstack([camera_params, port_param])

    return camera_params


def get_points_2d_df(points_csv_path):
    points_df = pd.read_csv(points_csv_path)

    points_2d_port_A = points_df[
        ["port_A", "bundle", "id", "x_A_raw", "y_A_raw"]
    ].rename(
        columns={
            "port_A": "camera",
            "bundle": "sync_index",
            "id": "point_id",
            "x_A_raw": "x_2d",
            "y_A_raw": "y_2d",
        }
    )

    points_2d_port_B = points_df[
        ["port_B", "bundle", "id", "x_B_raw", "y_B_raw"]
    ].rename(
        columns={
            "port_B": "camera",
            "bundle": "sync_index",
            "id": "point_id",
            "x_B_raw": "x_2d",
            "y_B_raw": "y_2d",
        }
    )

    points_2d_df = (
        pd.concat([points_2d_port_A, points_2d_port_B])
        .drop_duplicates()
        .sort_values(["sync_index", "point_id", "camera"])
        .rename(columns={"sync_index": "bundle"})
    )
    return points_2d_df


# get 3d points with indices to merge back into points_2d
def get_points_3d_df(points_csv_path):
    points_df = pd.read_csv(points_csv_path)
    points_3d_df = (
        points_df[["bundle", "id", "pair", "x_pos", "y_pos", "z_pos"]]
        .sort_values(["bundle", "id"])
        .groupby(["bundle", "id"])
        .agg({"x_pos": "mean", "y_pos": "mean", "z_pos": "mean", "pair": "size"})
        .rename(
            columns={"pair": "count", "x_pos": "x_3d", "y_pos": "y_3d", "z_pos": "z_3d"}
        )
        .reset_index()
        .reset_index()
        .rename(columns={"index": "index_3d", "id": "point_id"})
    )
    return points_3d_df


def get_point_data(points_csv_path:Path) -> PointData:

    points_2d_df = get_points_2d_df(points_csv_path)
    points_3d_df = get_points_3d_df(points_csv_path)

    merged_point_data = (
        points_2d_df.merge(points_3d_df, how="left", on=["bundle", "point_id"])
        .sort_values(["camera", "bundle", "point_id"])
        .dropna()
    )

    camera_indices = np.array(merged_point_data["camera"], dtype=np.int64)
    img = np.array(merged_point_data[["x_2d", "y_2d"]])
    obj_indices = np.array(merged_point_data["index_3d"], dtype=np.int64)
    obj = np.array(points_3d_df[["x_3d", "y_3d", "z_3d"]])

    return PointData(camera_indices, img, obj_indices, obj)

if __name__ == "__main__":
    
    import sys    
    from src.cameras.camera_array import CameraArrayBuilder
    repo = str(Path(__file__)).split("src")[0]

    print(repo)

    session_directory = Path(repo, "sessions", "iterative_adjustment")
    config_path = Path(session_directory, "config.toml")
    array_builder = CameraArrayBuilder(config_path)
    camera_array = array_builder.get_camera_array()

    params = get_camera_params(camera_array)
    print(params)