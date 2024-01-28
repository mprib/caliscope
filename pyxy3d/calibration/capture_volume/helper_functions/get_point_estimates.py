import caliscope.logger
logger = caliscope.logger.get(__name__)

import pandas as pd
import numpy as np
from caliscope.cameras.camera_array import CameraArray

from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.capture_volume.helper_functions.get_stereotriangulated_table import get_stereotriangulated_table
from pathlib import Path

def get_points_2d_df(stereotriangulated_table:pd.DataFrame):

    points_2d_port_A = stereotriangulated_table[
        ["port_A", "sync_index", "point_id", "x_A", "y_A"]
    ].rename(
        columns={
            "port_A": "camera",
            "sync_index": "sync_index",
            "point_id": "corner_id",
            "x_A": "x_2d",
            "y_A": "y_2d",
        }
    )

    points_2d_port_B = stereotriangulated_table[
        ["port_B", "sync_index", "point_id", "x_B", "y_B"]
    ].rename(
        columns={
            "port_B": "camera",
            "sync_index": "sync_index",
            "point_id": "corner_id",
            "x_B": "x_2d",
            "y_B": "y_2d",
        }
    )

    points_2d_df = (
        pd.concat([points_2d_port_A, points_2d_port_B])
        .drop_duplicates()
        .sort_values(["sync_index", "corner_id", "camera"])
        .rename(columns={"sync_index": "sync_index"})
    )
    return points_2d_df


# get 3d points with indices to merge back into points_2d
def get_points_3d_df(stereotriangulated_table):

    points_3d_df = (
        stereotriangulated_table[["sync_index", "point_id", "pair", "x_pos", "y_pos", "z_pos"]]
        .sort_values(["sync_index", "point_id"])
        .groupby(["sync_index", "point_id"])
        .agg({"x_pos": "mean", "y_pos": "mean", "z_pos": "mean", "pair": "size"})
        .rename(
            columns={"pair": "count", "x_pos": "x_3d", "y_pos": "y_3d", "z_pos": "z_3d"}
        )
        .reset_index()
        .reset_index()
        .rename(columns={"index": "index_3d", "point_id": "corner_id"})
    )
    return points_3d_df


def get_merged_2d_3d(stereotriangulated_table):
    """
    For each 2d point line, add in the estimated 3d point position
    """
    points_2d_df = get_points_2d_df(stereotriangulated_table)
    points_3d_df = get_points_3d_df(stereotriangulated_table)

    merged_point_data = (
        points_2d_df.merge(points_3d_df, how="left", on=["sync_index", "corner_id"])
        .sort_values(["camera", "sync_index", "corner_id"])
        .dropna()
    )

    return merged_point_data


def get_point_estimates(camera_array:CameraArray, point_data_path: Path) -> PointEstimates:
    """
    formats the triangulated_points.csv file into a PointEstimateData that has the 
    data structured in a way that is amenable to bundle adjustment
    """

    
    logger.info("Creating point history object based on camera_array and stereotriangulated_table")
    stereotriangulated_points = get_stereotriangulated_table(camera_array, point_data_path)
    
    
    points_3d_df = get_points_3d_df(stereotriangulated_points)
    merged_point_data = get_merged_2d_3d(stereotriangulated_points)

    camera_indices = np.array(merged_point_data["camera"], dtype=np.int64)
    img = np.array(merged_point_data[["x_2d", "y_2d"]])
    corner_id = np.array(merged_point_data["corner_id"], dtype=np.int64)
    obj_indices = np.array(merged_point_data["index_3d"], dtype=np.int64)
    sync_index = np.array(merged_point_data["sync_index"], dtype=np.int64)
    obj = np.array(points_3d_df[["x_3d", "y_3d", "z_3d"]])
    # obj_corner_id = np.array(points_3d_df[["corner_id"]])

    return PointEstimates(
        sync_indices=sync_index,
        camera_indices=camera_indices,
        point_id=corner_id,
        img=img,
        obj_indices=obj_indices,
        obj=obj,
        # obj_corner_id=obj_corner_id,
    )
