# caliscope/post_processing/point_data.py

from __future__ import annotations

from pathlib import Path
from typing import Type

import pandas as pd
import pandera as pa

# DataFrameModel is the correct base class for this style of schema
from pandera.typing import Series

# Forward-declare this type for use in method signatures
CameraArray = Type["CameraArray"]


class XYSchema(pa.DataFrameModel):
    """Pandera schema for validating 2D (x,y) point data."""

    sync_index: Series[int] = pa.Field(coerce=True)
    port: Series[int] = pa.Field(coerce=True)
    point_id: Series[int] = pa.Field(coerce=True)
    img_loc_x: Series[float] = pa.Field(coerce=True)
    img_loc_y: Series[float] = pa.Field(coerce=True)

    # Config class must inherit from the parent's Config
    class Config(pa.DataFrameModel.Config):
        # Allow other columns to pass through
        strict = False
        coerce = True


class XYZSchema(pa.DataFrameModel):
    """Pandera schema for validating 3D (x,y,z) point data."""

    sync_index: Series[int] = pa.Field(coerce=True)
    point_id: Series[int] = pa.Field(coerce=True)
    x_coord: Series[float] = pa.Field(coerce=True)
    y_coord: Series[float] = pa.Field(coerce=True)
    z_coord: Series[float] = pa.Field(coerce=True)

    # Config class must inherit from the parent's Config
    class Config(pa.DataFrameModel.Config):
        # Allow other columns to pass through
        strict = False
        coerce = True


class XYData:
    """
    A validated, immutable container for 2D (x,y) point data.

    This class ensures that the data it holds conforms to a predefined schema,
    preventing invalid or malformed data from propagating through the
    post-processing pipeline.

    Instances are created either from an existing DataFrame or directly
    from a CSV file. Once created, the internal DataFrame is read-only.
    """

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the XYData object by validating the input DataFrame.

        Args:
            df: A pandas DataFrame containing 2D point data.

        Raises:
            pa.errors.SchemaError: If the DataFrame fails validation.
        """
        self._df = XYSchema.validate(df)

    @property
    def df(self) -> pd.DataFrame:
        """Provides read-only access to the validated DataFrame."""
        return self._df.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> XYData:
        """
        Creates an XYData instance from a CSV file.

        Args:
            path: The path to the CSV file.

        Returns:
            A new instance of XYData.
        """
        df = pd.read_csv(path)
        return cls(df)

    def fill_gaps(self, max_gap_size: int) -> XYData:
        """
        Fills missing data points up to a specified gap size.

        This method will contain the gap-filling logic. Currently a placeholder.

        Args:
            max_gap_size: The maximum number of consecutive missing frames to fill.

        Returns:
            A new XYData instance with gaps filled.
        """
        # To be implemented in Step 3
        raise NotImplementedError("Gap filling logic will be implemented here.")

    def triangulate(self, camera_array: CameraArray) -> XYZData:
        """
        Triangulates 2D points to create 3D points.

        This method will contain the triangulation logic. Currently a placeholder.

        Args:
            camera_array: A CameraArray object containing calibration info.

        Returns:
            A new XYZData instance containing the 3D point data.
        """
        # To be implemented in Step 3
        raise NotImplementedError("Triangulation logic will be implemented here.")


class XYZData:
    """
    A validated, immutable container for 3D (x,y,z) point data.

    This class ensures that the data it holds conforms to a predefined schema.
    It follows the same principles of validation and immutability as XYData.
    """

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the XYZData object by validating the input DataFrame.

        Args:
            df: A pandas DataFrame containing 3D point data.

        Raises:
            pa.errors.SchemaError: If the DataFrame fails validation.
        """
        self._df = XYZSchema.validate(df)

    @property
    def df(self) -> pd.DataFrame:
        """Provides read-only access to the validated DataFrame."""
        return self._df.copy()

    @classmethod
    def from_csv(cls, path: str | Path) -> XYZData:
        """
        Creates an XYZData instance from a CSV file.

        Args:
            path: The path to the CSV file.

        Returns:
            A new instance of XYZData.
        """
        df = pd.read_csv(path)
        return cls(df)

    def fill_gaps(self, max_gap_size: int) -> XYZData:
        """
        Fills missing data points up to a specified gap size.

        This method will contain the gap-filling logic. Currently a placeholder.

        Args:
            max_gap_size: The maximum number of consecutive missing frames to fill.

        Returns:
            A new XYZData instance with gaps filled.
        """
        # To be implemented in Step 3
        raise NotImplementedError("Gap filling logic will be implemented here.")

    def smooth(self, cutoff_freq: int) -> XYZData:
        """
        Applies a smoothing filter to the 3D data.

        This method will contain the smoothing logic. Currently a placeholder.

        Args:
            cutoff_freq: The cutoff frequency for the smoothing filter.

        Returns:
            A new XYZData instance with smoothed data.
        """
        # To be implemented in Step 3
        raise NotImplementedError("Smoothing logic will be implemented here.")
