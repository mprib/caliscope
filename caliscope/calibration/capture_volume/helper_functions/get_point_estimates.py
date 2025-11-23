# caliscope/calibration/capture_volume/helper_functions/get_point_estimates.py

import logging
import numpy as np

from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.point_data import ImagePoints, XYZData

logger = logging.getLogger(__name__)


def create_point_estimates_from_stereopairs(camera_array: CameraArray, image_points: ImagePoints) -> PointEstimates:
    """
    Create PointEstimates from stereo triangulated points.

    Replaces legacy pipeline with direct conversion from ImagePoints.triangulate().
    """
    logger.info("Creating point estimates using ImagePoints.triangulate()")

    # Use the clean triangulation implementation
    xyz_data: XYZData = image_points.triangulate(camera_array)

    if xyz_data.df.empty:
        logger.warning("No points triangulated")
        return PointEstimates(
            sync_indices=np.array([], dtype=np.int64),
            camera_indices=np.array([], dtype=np.int64),
            point_id=np.array([], dtype=np.int64),
            img=np.array([], dtype=np.float32).reshape(0, 2),
            obj_indices=np.array([], dtype=np.int64),
            obj=np.array([], dtype=np.float32).reshape(0, 3),
        )

    return _convert_xyz_to_point_estimates(xyz_data, image_points, camera_array)


def _convert_xyz_to_point_estimates(
    xyz_data: XYZData, image_points: ImagePoints, camera_array: CameraArray
) -> PointEstimates:
    """
    Convert XYZData + ImagePoints to PointEstimates format needed for bundle adjustment.
    """
    xyz_df = xyz_data.df
    xy_df = image_points.df

    # Create mapping from (sync_index, point_id) to xyz_index
    xyz_df = xyz_df.reset_index().rename(columns={"index": "xyz_index"})
    {(row["sync_index"], row["point_id"]): row["xyz_index"] for _, row in xyz_df.iterrows()}

    # Merge 2D and 3D data
    merged = xy_df.merge(
        xyz_df[["sync_index", "point_id", "xyz_index", "x_coord", "y_coord", "z_coord"]],
        on=["sync_index", "point_id"],
        how="inner",
    )

    # Filter to posed cameras only
    posed_ports = list(camera_array.posed_port_to_index.keys())
    merged = merged[merged["port"].isin(posed_ports)]

    if merged.empty:
        logger.warning("No merged 2D-3D observations after filtering")
        return PointEstimates(
            sync_indices=np.array([], dtype=np.int64),
            camera_indices=np.array([], dtype=np.int64),
            point_id=np.array([], dtype=np.int64),
            img=np.array([], dtype=np.float32).reshape(0, 2),
            obj_indices=np.array([], dtype=np.int64),
            obj=np.array([], dtype=np.float32).reshape(0, 3),
        )

    # Map ports to camera indices
    merged["camera_index"] = merged["port"].map(camera_array.posed_port_to_index)

    # Extract arrays
    sync_indices = merged["sync_index"].to_numpy(dtype=np.int64)
    camera_indices = merged["camera_index"].to_numpy(dtype=np.int64)
    point_ids = merged["point_id"].to_numpy(dtype=np.int64)
    img = merged[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32)
    obj_indices = merged["xyz_index"].to_numpy(dtype=np.int64)

    # Get unique 3D points
    obj = xyz_df[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)

    # Validate consistency
    assert len(camera_indices) == len(img) == len(obj_indices), "Mismatch in 2D data array lengths"
    if len(obj_indices) > 0:
        assert obj_indices.max() < obj.shape[0], "CRITICAL: obj_indices contains an out-of-bounds index"

    logger.info(f"Successfully created PointEstimates: {obj.shape[0]} 3D points and {img.shape[0]} 2D observations")

    return PointEstimates(
        sync_indices=sync_indices,
        camera_indices=camera_indices,
        point_id=point_ids,
        img=img,
        obj_indices=obj_indices,
        obj=obj,
    )
