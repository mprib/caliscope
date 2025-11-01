# caliscope/post_processing/point_data.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

# Forward-declare this type for use in method signatures
CameraArray = Type["CameraArray"]


class XYSchema(pa.DataFrameModel):
    """Pandera schema for validating 2D (x,y) point data."""

    sync_index: Series[int] = pa.Field(coerce=True)
    port: Series[int] = pa.Field(coerce=True)
    point_id: Series[int] = pa.Field(coerce=True)
    img_loc_x: Series[float] = pa.Field(coerce=True)
    img_loc_y: Series[float] = pa.Field(coerce=True)

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True


class XYZSchema(pa.DataFrameModel):
    """Pandera schema for validating 3D (x,y,z) point data."""

    sync_index: Series[int] = pa.Field(coerce=True)
    point_id: Series[int] = pa.Field(coerce=True)
    x_coord: Series[float] = pa.Field(coerce=True)
    y_coord: Series[float] = pa.Field(coerce=True)
    z_coord: Series[float] = pa.Field(coerce=True)

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True


class XYData:
    """A validated, immutable container for 2D (x,y) point data."""

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self._df = XYSchema.validate(df)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> XYData:
        df = pd.read_csv(path)
        return cls(df)

    def fill_gaps(self, max_gap_size: int = 3) -> XYData:
        """
        Fills missing data points up to a specified gap size via linear interpolation.

        This method operates on each point from each camera independently.

        Args:
            max_gap_size: The maximum number of consecutive missing frames to fill.

        Returns:
            A new XYData instance with gaps filled.
        """
        xy_filled = pd.DataFrame()
        index_key = "sync_index"
        last_port = -1

        base_df = self.df  # Use the validated, copied dataframe
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

            # Interpolate values. Other columns like frame_time will also be interpolated if present.
            for col in ["img_loc_x", "img_loc_y", "frame_time"]:
                if col in merged.columns:
                    merged[col] = merged[col].interpolate(method="linear", limit=max_gap_size)

            xy_filled = pd.concat([xy_filled, merged])

        logger.info("(x,y) gap filling complete")
        return XYData(xy_filled.dropna(subset=["img_loc_x"]))

    def triangulate(self, camera_array: CameraArray) -> XYZData:
        """
        Triangulates 2D points to create 3D points.
        This method will contain the triangulation logic. Currently a placeholder.
        """
        raise NotImplementedError("Triangulation logic will be implemented here.")


class XYZData:
    """A validated, immutable container for 3D (x,y,z) point data."""

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self._df = XYZSchema.validate(df)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> XYZData:
        df = pd.read_csv(path)
        return cls(df)

    def fill_gaps(self, max_gap_size: int = 3) -> XYZData:
        """
        Fills missing 3D data points up to a specified gap size via linear interpolation.

        This method operates on each 3D point independently.

        Args:
            max_gap_size: The maximum number of consecutive missing frames to fill.

        Returns:
            A new XYZData instance with gaps filled.
        """
        xyz_filled = pd.DataFrame()
        base_df = self.df  # Use the validated, copied dataframe

        for point_id, group in base_df.groupby("point_id"):
            group = group.sort_values("sync_index")
            all_frames = pd.DataFrame(
                {"sync_index": np.arange(group["sync_index"].min(), group["sync_index"].max() + 1)}
            )
            all_frames["point_id"] = point_id

            merged = pd.merge(all_frames, group, on=["point_id", "sync_index"], how="left")
            merged["gap_size"] = (
                merged["x_coord"].isnull().astype(int).groupby((merged["x_coord"].notnull()).cumsum()).cumsum()
            )
            merged = merged[merged["gap_size"] <= max_gap_size]

            for col in ["x_coord", "y_coord", "z_coord"]:
                if col in merged.columns:
                    merged[col] = merged[col].interpolate(method="linear", limit=max_gap_size)

            xyz_filled = pd.concat([xyz_filled, merged])

        return XYZData(xyz_filled.dropna(subset=["x_coord"]))

    def smooth(self, fps: float, cutoff_freq: float, order: int = 2) -> XYZData:
        """
        Applies a zero-phase Butterworth filter to the 3D data.

        Args:
            fps: The framerate of the data capture.
            cutoff_freq: The cutoff frequency for the low-pass filter.
            order: The order of the Butterworth filter. Defaults to 2.

        Returns:
            A new XYZData instance with smoothed data.
        """
        b, a = butter(order, cutoff_freq, btype="low", fs=fps)

        base_df = self.df
        xyz_filtered = base_df.copy()

        for point_id, group in base_df.groupby("point_id"):
            # Skip smoothing if there aren't enough data points to apply the filter
            if group.shape[0] > 3 * order:
                xyz_filtered.loc[group.index, "x_coord"] = filtfilt(b, a, group["x_coord"])
                xyz_filtered.loc[group.index, "y_coord"] = filtfilt(b, a, group["y_coord"])
                xyz_filtered.loc[group.index, "z_coord"] = filtfilt(b, a, group["z_coord"])

        return XYZData(xyz_filtered)
