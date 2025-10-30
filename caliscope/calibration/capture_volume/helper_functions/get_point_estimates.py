import logging
from pathlib import Path

import numpy as np
import pandas as pd

from caliscope.calibration.capture_volume.helper_functions.get_stereotriangulated_table import (
    get_stereotriangulated_table,
)
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.cameras.camera_array import CameraArray

logger = logging.getLogger(__name__)


def get_points_2d_df(stereotriangulated_table: pd.DataFrame) -> pd.DataFrame:
    """Extracts and formats 2D point data from the triangulated table."""
    points_2d_port_A = stereotriangulated_table[["port_A", "sync_index", "point_id", "x_A", "y_A"]].rename(
        columns={"port_A": "camera", "x_A": "x_2d", "y_A": "y_2d"}
    )

    points_2d_port_B = stereotriangulated_table[["port_B", "sync_index", "point_id", "x_B", "y_B"]].rename(
        columns={"port_B": "camera", "x_B": "x_2d", "y_B": "y_2d"}
    )

    points_2d_df = (
        pd.concat([points_2d_port_A, points_2d_port_B])
        .drop_duplicates()
        .sort_values(["sync_index", "point_id", "camera"])
    )
    return points_2d_df


def get_points_3d_df(stereotriangulated_table: pd.DataFrame) -> pd.DataFrame:
    """Extracts, averages, and indexes 3D point data from the triangulated table."""
    points_3d_df = (
        stereotriangulated_table[["sync_index", "point_id", "pair", "x_pos", "y_pos", "z_pos"]]
        .sort_values(["sync_index", "point_id"])
        .groupby(["sync_index", "point_id"])
        .agg({"x_pos": "mean", "y_pos": "mean", "z_pos": "mean", "pair": "size"})
        .rename(columns={"pair": "count", "x_pos": "x_3d", "y_pos": "y_3d", "z_pos": "z_3d"})
        .reset_index()
        .reset_index()
        .rename(columns={"index": "index_3d"})
    )
    return points_3d_df


def get_merged_2d_3d(stereotriangulated_table: pd.DataFrame) -> pd.DataFrame:
    """Merges 2D and 3D point dataframes, associating each 2D observation with its 3D estimate."""
    points_2d_df = get_points_2d_df(stereotriangulated_table)
    points_3d_df = get_points_3d_df(stereotriangulated_table)

    merged_point_data = (
        points_2d_df.merge(points_3d_df, how="left", on=["sync_index", "point_id"])
        .sort_values(["camera", "sync_index", "point_id"])
        .dropna()
    )

    return merged_point_data


def create_point_estimates_from_stereopairs(camera_array: CameraArray, point_data_path: Path) -> PointEstimates:
    """
    Stereotriangulates data to generate initial x,y,z estimates and formats the
    data into a PointEstimates object suitable for bundle adjustment.
    """
    logger.info("Creating point estimates based on camera_array and stereotriangulated_table")
    stereotriangulated_points = get_stereotriangulated_table(camera_array, point_data_path)

    points_3d_df = get_points_3d_df(stereotriangulated_points)
    merged_point_data = get_merged_2d_3d(stereotriangulated_points)
    logger.info(f"Initial data loaded: {len(merged_point_data)} 2D observations and {len(points_3d_df)} 3D points.")

    # Get the dictionary that maps active port numbers to their zero-based index
    port_to_index_map = camera_array.posed_port_to_index
    posed_ports = list(port_to_index_map.keys())

    # Filter the merged data to only include observations from posed cameras.
    # This removes any 2D points associated with the unlinked camera
    logger.info(f"Filtering point data to include only posed cameras: {posed_ports}")
    filtered_merged_data = merged_point_data[merged_point_data["camera"].isin(posed_ports)]
    logger.info(f"Retained {len(filtered_merged_data)} 2D observations after filtering for posed cameras.")

    # Map the camera port numbers (e.g., 1, 2, 3, 6) to the correct
    # zero-based indices for optimization (e.g., 0, 1, 2, 3).
    camera_indices = filtered_merged_data["camera"].map(port_to_index_map).to_numpy(dtype=np.int64)

    # After filtering 2D observations, some 3D points may no longer be referenced.
    # We must prune the 3D points list and re-index obj_indices to ensure consistency.

    # 1. Identify the unique 3D point indices that are still in use.
    unique_obj_indices_original = filtered_merged_data["index_3d"].unique()
    logger.info(
        f"Identified {len(unique_obj_indices_original)} unique 3D points referenced by the remaining 2D observations."
    )

    # 2. Create a new, compact `obj` array containing only these referenced points.
    points_3d_df_pruned = points_3d_df[points_3d_df["index_3d"].isin(unique_obj_indices_original)]
    obj = np.array(points_3d_df_pruned[["x_3d", "y_3d", "z_3d"]])

    pruned_count = len(points_3d_df) - len(points_3d_df_pruned)
    if pruned_count > 0:
        logger.info(f"Pruned {pruned_count} orphaned 3D points that are no longer referenced.")

    # 3. Create a mapping from the original 'index_3d' to the new, dense index.
    # This is crucial for creating the new obj_indices array.
    old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(points_3d_df_pruned["index_3d"])}

    # 4. Apply the mapping to create the final, consistent obj_indices.
    obj_indices = filtered_merged_data["index_3d"].map(old_to_new_map).to_numpy(dtype=np.int64)

    # Extract remaining data from the correctly filtered and now consistent dataframe
    img = np.array(filtered_merged_data[["x_2d", "y_2d"]])
    point_id = np.array(filtered_merged_data["point_id"], dtype=np.int64)
    sync_index = np.array(filtered_merged_data["sync_index"], dtype=np.int64)

    # --- Final Consistency Checks ---
    assert len(camera_indices) == len(img) == len(obj_indices), "Mismatch in 2D data array lengths."
    if len(obj_indices) > 0:
        assert obj_indices.max() < obj.shape[0], "CRITICAL: obj_indices contains an out-of-bounds index."
    if len(obj_indices) > 0:
        assert (
            np.unique(obj_indices).size == obj.shape[0]
        ), "Mismatch between unique object indices and number of 3D points."

    logger.info(
        f"Successfully created consistent PointEstimates: {obj.shape[0]} 3D points and {img.shape[0]} 2D observations."
    )

    return PointEstimates(
        sync_indices=sync_index,
        camera_indices=camera_indices,
        point_id=point_id,
        img=img,
        obj_indices=obj_indices,
        obj=obj,
    )
