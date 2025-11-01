# caliscope/triangulate/triangulation.py

import logging
from pathlib import Path

import pandas as pd

from caliscope.configurator import Configurator
from caliscope.post_processing.point_data import XYData

logger = logging.getLogger(__name__)


def triangulate_from_files(config_path: Path, xy_path: Path, output_path: Path = None) -> pd.DataFrame:
    """
    Triangulate 2D points to 3D using camera calibration from config.toml

    Parameters
    ----------
    config_path : Path
        Path to config.toml containing camera calibration
    xy_path : Path
        Path to CSV file with 2D point data:
        - sync_index: Temporal index for synchronization
        - port: Camera ID/port
        - point_id: ID of the tracked point
        - img_loc_x: X-coordinate in image
        - img_loc_y: Y-coordinate in image

    output_path : Path, optional
        Path where triangulated 3D points will be saved
        If not provided, results are returned but not saved

    Returns
    -------
    pd.DataFrame
        DataFrame containing triangulated 3D points with columns:
        - sync_index: Temporal index for synchronization
        - point_id: ID of the tracked point
        - x_coord: X-coordinate in 3D space
        - y_coord: Y-coordinate in 3D space
        - z_coord: Z-coordinate in 3D space
    """
    logger.info(f"Loading configuration from {config_path}")
    config = Configurator(config_path.parent)

    logger.info("Loading camera array")
    camera_array = config.get_camera_array()

    logger.info(f"Loading 2D points from {xy_path} into a validated XYData object")
    xy_data = XYData.from_csv(xy_path)

    logger.info("Beginning triangulation...")
    xyz_data = xy_data.triangulate(camera_array)

    if output_path:
        logger.info(f"Saving triangulated points to {output_path}")
        xyz_data.df.to_csv(output_path, index=False)

    logger.info("Triangulation complete")
    return xyz_data.df
