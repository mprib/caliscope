# caliscope/post_processing/point_data.py

from __future__ import annotations
import logging
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import pandera.pandas as pa
from numba import jit
from numba.typed import Dict, List
from pandera.typing import Series
from scipy.signal import butter, filtfilt
from caliscope.cameras.camera_array import CameraArray
from caliscope.calibration.capture_volume.point_estimates import PointEstimates

# Add to existing imports at top of file
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


#####################################################################################
# The following code is adapted from the `Anipose` project,
# in particular the `triangulate_simple` function of `aniposelib`
# Original author:  Lili Karashchuk
# Project: https://github.com/lambdaloop/aniposelib/
# Original Source Code : https://github.com/lambdaloop/aniposelib/blob/d03b485c4e178d7cff076e9fe1ac36837db49158/aniposelib/cameras.py#L21
# This code is licensed under the BSD 2-Clause License
# BSD 2-Clause License

# Copyright (c) 2019, Lili Karashchuk
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@jit(nopython=True, cache=True)
def triangulate_sync_index(
    projection_matrices: Dict, current_camera_indices: np.ndarray, current_point_id: np.ndarray, current_img: np.ndarray
):
    """A more optimized Numba function to triangulate points for a single sync_index."""
    point_indices_xyz = List()
    obj_xyz = List()

    # Exit early if there's not enough data to form a pair
    if len(current_point_id) < 2:
        return point_indices_xyz, obj_xyz

    # Sort by point_id to group all observations of the same point together
    sort_indices = np.argsort(current_point_id)
    sorted_points = current_point_id[sort_indices]
    sorted_cams = current_camera_indices[sort_indices]
    sorted_img = current_img[sort_indices]

    group_start = 0
    # Iterate through the sorted arrays to find groups of points
    for i in range(1, len(sorted_points)):
        # A new group starts when the point_id changes
        if sorted_points[i] != sorted_points[group_start]:
            # Process the previous group if it has enough views
            if i - group_start > 1:
                point = sorted_points[group_start]
                points_xy = sorted_img[group_start:i]
                camera_ids = sorted_cams[group_start:i]
                num_cams = len(camera_ids)

                A = np.zeros((num_cams * 2, 4))
                for j in range(num_cams):
                    x, y = points_xy[j]
                    P = projection_matrices[camera_ids[j]]
                    A[(j * 2) : (j * 2 + 1)] = x * P[2] - P[0]
                    A[(j * 2 + 1) : (j * 2 + 2)] = y * P[2] - P[1]

                u, s, vh = np.linalg.svd(A, full_matrices=True)
                point_xyzw = vh[-1]
                point_xyz = point_xyzw[:3] / point_xyzw[3]
                point_indices_xyz.append(point)
                obj_xyz.append(point_xyz)

            # Start the new group
            group_start = i

    # Process the final group after the loop finishes
    if len(sorted_points) - group_start > 1:
        point = sorted_points[group_start]
        # Slicing to the end is implicit
        points_xy = sorted_img[group_start:]
        camera_ids = sorted_cams[group_start:]
        # ... (SVD logic repeated for the last group - could be refactored)
        num_cams = len(camera_ids)
        A = np.zeros((num_cams * 2, 4))
        for j in range(num_cams):
            x, y = points_xy[j]
            P = projection_matrices[camera_ids[j]]
            A[(j * 2) : (j * 2 + 1)] = x * P[2] - P[0]
            A[(j * 2 + 1) : (j * 2 + 2)] = y * P[2] - P[1]

        u, s, vh = np.linalg.svd(A, full_matrices=True)
        point_xyzw = vh[-1]
        point_xyz = point_xyzw[:3] / point_xyzw[3]
        point_indices_xyz.append(point)
        obj_xyz.append(point_xyz)

    return point_indices_xyz, obj_xyz


############################################################################################


def _undistort_batch(xy_df: pd.DataFrame, camera_array: CameraArray) -> pd.DataFrame:
    """Module-private helper to undistort all points in a DataFrame."""
    undistorted_points = []
    for port, camera in camera_array.cameras.items():
        subset_xy = xy_df.query(f"port == {port}").copy()
        if not subset_xy.empty:
            points = np.vstack([subset_xy["img_loc_x"], subset_xy["img_loc_y"]]).T
            undistorted_xy = camera.undistort_points(points)
            subset_xy["img_loc_undistort_x"] = undistorted_xy[:, 0]
            subset_xy["img_loc_undistort_y"] = undistorted_xy[:, 1]
            undistorted_points.append(subset_xy)

    if not undistorted_points:
        return pd.DataFrame()

    xy_undistorted_df = pd.concat(undistorted_points)
    return xy_undistorted_df


class ImagePointSchema(pa.DataFrameModel):
    """Pandera schema for validating 2D (x,y) point data."""

    sync_index: Series[int] = pa.Field(coerce=True)
    port: Series[int] = pa.Field(coerce=True)
    point_id: Series[int] = pa.Field(coerce=True)
    img_loc_x: Series[float] = pa.Field(coerce=True)
    img_loc_y: Series[float] = pa.Field(coerce=True)

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True


class WorldPointSchema(pa.DataFrameModel):
    """Pandera schema for validating 3D (x,y,z) point data."""

    sync_index: Series[int] = pa.Field(coerce=True)
    point_id: Series[int] = pa.Field(coerce=True)
    x_coord: Series[float] = pa.Field(coerce=True)
    y_coord: Series[float] = pa.Field(coerce=True)
    z_coord: Series[float] = pa.Field(coerce=True)

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True


class ImagePoints:
    """A validated, immutable container for 2D (x,y) point data."""

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self._df = ImagePointSchema.validate(df)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> ImagePoints:
        df = pd.read_csv(path)

        return cls(ImagePointSchema.validate(df))

    def fill_gaps(self, max_gap_size: int = 3) -> ImagePoints:
        xy_filled = pd.DataFrame()
        index_key = "sync_index"
        last_port = -1
        base_df = self.df
        for (port, point_id), group in base_df.groupby(["port", "point_id"]):
            if last_port != port:
                logger.info(
                    f"Gap filling for (x,y) data from port {port}. "
                    f"Filling gaps that are {max_gap_size} frames or less..."
                )
            last_port = port
            group = group.sort_values(index_key)
            all_frames = pd.DataFrame({index_key: np.arange(group[index_key].min(), group[index_key].max() + 1)})
            all_frames["port"] = port
            all_frames["point_id"] = point_id
            merged = pd.merge(all_frames, group, on=["port", "point_id", index_key], how="left")
            merged["gap_size"] = (
                merged["img_loc_x"].isnull().astype(int).groupby((merged["img_loc_x"].notnull()).cumsum()).cumsum()
            )
            merged = merged[merged["gap_size"] <= max_gap_size]
            for col in ["img_loc_x", "img_loc_y", "frame_time"]:
                if col in merged.columns:
                    merged[col] = merged[col].interpolate(method="linear", limit=max_gap_size)
            xy_filled = pd.concat([xy_filled, merged])
        logger.info("(x,y) gap filling complete")
        return ImagePoints(xy_filled.dropna(subset=["img_loc_x"]))

    def triangulate(self, camera_array: CameraArray) -> WorldPoints:
        """
        Triangulates 2D points to create 3D points using the provided CameraArray.
        The input 2D points are undistorted as part of this process.
        """
        xy_df = self.df
        if xy_df.empty:
            return WorldPoints(pd.DataFrame(columns=list(WorldPointSchema.to_schema().columns.keys())))

        # Only process cameras that are both in data AND posed
        ports_in_data = xy_df["port"].unique()
        posed_ports = list(camera_array.posed_port_to_index.keys())
        valid_ports = [p for p in ports_in_data if p in posed_ports]

        if not valid_ports:
            logger.warning("No cameras in data have extrinsics for triangulation")
            return WorldPoints(pd.DataFrame(columns=list(WorldPointSchema.to_schema().columns.keys())))

        # Assemble numba compatible dictionary for projection matrices
        # This already filters to posed cameras
        normalized_projection_matrices = camera_array.normalized_projection_matrices

        # Undistort all image points before triangulation
        undistorted_xy = _undistort_batch(xy_df, camera_array)

        xyz_data = {
            "sync_index": [],
            "point_id": [],
            "x_coord": [],
            "y_coord": [],
            "z_coord": [],
        }

        # sync_index_max = xy_df["sync_index"].max()
        start = time()
        last_log_update = int(start)

        logger.info("About to begin triangulation...due to jit, first round of calculations may take a moment.")

        # Only iterate over sync indices that have data from valid ports
        valid_sync_indices = undistorted_xy[undistorted_xy["port"].isin(valid_ports)]["sync_index"].unique()

        sync_index_counter = 0
        total_sync_indices = len(valid_sync_indices)

        for index in valid_sync_indices:
            sync_index_counter += 1  # used for tracking progress
            active_index = undistorted_xy["sync_index"] == index
            # Filter to valid ports for this sync index
            index_data = undistorted_xy[active_index & undistorted_xy["port"].isin(valid_ports)]

            if index_data.empty:
                continue

            port = index_data["port"].to_numpy()
            point_ids = index_data["point_id"].to_numpy()
            img_loc_x = index_data["img_loc_undistort_x"].to_numpy()
            img_loc_y = index_data["img_loc_undistort_y"].to_numpy()
            raw_xy = np.vstack([img_loc_x, img_loc_y]).T

            point_id_xyz, points_xyz = triangulate_sync_index(normalized_projection_matrices, port, point_ids, raw_xy)

            if len(point_id_xyz) > 0:
                xyz_data["sync_index"].extend([index] * len(point_id_xyz))
                xyz_data["point_id"].extend(point_id_xyz)
                points_xyz_arr = np.array(points_xyz)
                xyz_data["x_coord"].extend(points_xyz_arr[:, 0].tolist())
                xyz_data["y_coord"].extend(points_xyz_arr[:, 1].tolist())
                xyz_data["z_coord"].extend(points_xyz_arr[:, 2].tolist())

            if int(time()) - last_log_update >= 1:
                percent_complete = int(100 * (sync_index_counter / total_sync_indices))
                logger.info(f"Triangulation of (x,y) point estimates is {percent_complete}% complete")
                last_log_update = int(time())

        xyz_df = pd.DataFrame(xyz_data)
        return WorldPoints(xyz_df)


@dataclass(frozen=True)
class WorldPoints:
    """A validated, immutable container for 3D (x,y,z) point data."""

    _df: pd.DataFrame

    def __post_init__(self):
        # Validate schema
        object.__setattr__(self, "_df", WorldPointSchema.validate(self._df))

    @property
    def df(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame to maintain immutability."""
        return self._df.copy()

    @property
    def points(self) -> np.ndarray:
        """Return Nx3 numpy array of coordinates."""
        return self._df[["x_coord", "y_coord", "z_coord"]].values

    # Keep existing methods for backward compatibility
    @property
    def source_image_points(self) -> Optional[ImagePoints]:
        return self._source_image_points

    @property
    def camera_array(self) -> Optional[CameraArray]:
        return self._camera_array

    def to_point_estimates(self, image_points: ImagePoints, camera_array: CameraArray) -> PointEstimates:
        xyz_df = self.df
        xy_df = image_points.df

        # Create explicit mapping in the dataframe to track indices.
        # We reset index to ensure we have a column "xyz_index" corresponding to the row number in xyz_df
        xyz_df = xyz_df.reset_index(drop=True)
        xyz_df["xyz_index"] = xyz_df.index

        # Merge 2D and 3D data
        merged = xy_df.merge(
            xyz_df[["sync_index", "point_id", "xyz_index"]],  # We don't strictly need coords here
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

        # =========================================================================
        # Prune orphaned 3D points
        # =========================================================================

        # 1. Identify which xyz_indices are actually used after filtering cameras
        used_xyz_indices = merged["xyz_index"].unique()
        used_xyz_indices.sort()  # Ensure deterministic order

        # 2. Extract ONLY the used 3D points from the master list
        obj = xyz_df.loc[used_xyz_indices, ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)

        # 3. Create a map from the old (large) index to the new (compact) index
        old_to_new_map = {old: new for new, old in enumerate(used_xyz_indices)}

        # 4. Update the pointers in the merged dataframe to reflect the new compact array
        merged["obj_index_pruned"] = merged["xyz_index"].map(old_to_new_map)
        obj_indices = merged["obj_index_pruned"].to_numpy(dtype=np.int64)

        # =========================================================================

        # Map ports to camera indices
        merged["camera_index"] = merged["port"].map(camera_array.posed_port_to_index)

        # Extract arrays
        sync_indices = merged["sync_index"].to_numpy(dtype=np.int64)
        camera_indices = merged["camera_index"].to_numpy(dtype=np.int64)
        point_ids = merged["point_id"].to_numpy(dtype=np.int64)
        img = merged[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32)

        # Validate consistency
        assert len(camera_indices) == len(img) == len(obj_indices), "Mismatch in 2D data array lengths"
        if len(obj_indices) > 0:
            assert obj_indices.max() < obj.shape[0], "CRITICAL: obj_indices contains an out-of-bounds index"
            # The key check for the hang:
            assert np.unique(obj_indices).size == obj.shape[0], (
                "CRITICAL: Orphaned 3D points detected! Optimizer will hang."
            )

        logger.info(f"Successfully created PointEstimates: {obj.shape[0]} 3D points and {img.shape[0]} 2D observations")

        return PointEstimates(
            sync_indices=sync_indices,
            camera_indices=camera_indices,
            point_id=point_ids,
            img=img,
            obj_indices=obj_indices,
            obj=obj,
        )

    def fill_gaps(self, max_gap_size: int = 3) -> WorldPoints:
        """Fill gaps in 3D point trajectories."""
        xyz_filled = pd.DataFrame()
        base_df = self.df

        for point_id, group in base_df.groupby("point_id"):
            group = group.sort_values("sync_index")
            all_frames = pd.DataFrame(
                {"sync_index": np.arange(group["sync_index"].min(), group["sync_index"].max() + 1)}
            )
            all_frames["point_id"] = point_id
            merged = pd.merge(all_frames, group, on=["point_id", "sync_index"], how="left")

            # Calculate gap size
            merged["gap_size"] = (
                merged["x_coord"].isnull().astype(int).groupby((merged["x_coord"].notnull()).cumsum()).cumsum()
            )
            merged = merged[merged["gap_size"] <= max_gap_size]

            # Interpolate coordinates
            for col in ["x_coord", "y_coord", "z_coord"]:
                if col in merged.columns:
                    merged[col] = merged[col].interpolate(method="linear", limit=max_gap_size)

            xyz_filled = pd.concat([xyz_filled, merged])

        # Return new WorldPoints instance (immutable pattern)
        return WorldPoints(xyz_filled.dropna(subset=["x_coord"]))

    def smooth(self, fps: float, cutoff_freq: float, order: int = 2) -> WorldPoints:
        """Apply Butterworth filter to smooth 3D trajectories."""
        b, a = butter(order, cutoff_freq, btype="low", fs=fps)
        base_df = self.df
        xyz_filtered = base_df.copy()

        for point_id, group in base_df.groupby("point_id"):
            if group.shape[0] > 3 * order:
                xyz_filtered.loc[group.index, "x_coord"] = filtfilt(b, a, group["x_coord"])
                xyz_filtered.loc[group.index, "y_coord"] = filtfilt(b, a, group["y_coord"])
                xyz_filtered.loc[group.index, "z_coord"] = filtfilt(b, a, group["z_coord"])

        # Return new WorldPoints instance (immutable pattern)
        return WorldPoints(xyz_filtered)

    @classmethod
    def from_point_estimates(cls, point_estimates: PointEstimates) -> WorldPoints:
        """
        Reconstruct WorldPoints from optimized PointEstimates.

        Creates WorldPoints by extracting unique 3D points and their identifiers
        from the optimized PointEstimates structure. This is used when loading
        previously optimized calibration data where only the filtered subset
        of observations is available.
        """
        # Get unique object indices and their first occurrence
        unique_obj_indices = np.unique(point_estimates.obj_indices)

        # For each unique 3D point, find its first observation to get identifiers
        world_data = {
            "sync_index": [],
            "point_id": [],
            "x_coord": [],
            "y_coord": [],
            "z_coord": [],
        }

        for obj_idx in unique_obj_indices:
            # Find first observation of this 3D point
            first_obs_idx = np.where(point_estimates.obj_indices == obj_idx)[0][0]

            world_data["sync_index"].append(point_estimates.sync_indices[first_obs_idx])
            world_data["point_id"].append(point_estimates.point_id[first_obs_idx])

            # Get 3D coordinates
            coords = point_estimates.obj[obj_idx]
            world_data["x_coord"].append(coords[0])
            world_data["y_coord"].append(coords[1])
            world_data["z_coord"].append(coords[2])

        world_df = pd.DataFrame(world_data)
        return cls(world_df)
