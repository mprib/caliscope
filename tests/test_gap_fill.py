
#%%
import caliscope.logger


import pandas as pd
import numpy as np

from pathlib import Path
from caliscope.trackers.tracker_enum import TrackerEnum
from caliscope.post_processing.gap_filling import gap_fill_xy, gap_fill_xyz
from caliscope import __root__
from caliscope.helper import copy_contents
logger = caliscope.logger.get(__name__)

original_recording_directory = Path(__root__, "tests", "reference", "base_data")
tracker_enum = TrackerEnum.HOLISTIC_OPENSIM

recording_directory = Path(original_recording_directory.parent.parent, "reference_delete", "base_data")
copy_contents(original_recording_directory, recording_directory)

def test_gap_fill_xy():

    # Load the data
    xy_all_base_path = Path(recording_directory,tracker_enum.name, f"xy_{tracker_enum.name}.csv")
    logger.info(f"Reading in raw xy data located at {xy_all_base_path}")
    xy_all_base = pd.read_csv(xy_all_base_path)
    base_length = xy_all_base.shape[0]
    xy_all_filled = gap_fill_xy(xy_all_base)
    filled_length = xy_all_filled.shape[0]

    xy_filled_path = Path(xy_all_base_path.parent, f"xy_{tracker_enum.name}_filled.csv")
    xy_all_filled.to_csv(xy_filled_path, index=False)

    # a coupLe basic assertions
    assert base_length < filled_length
    assert xy_all_filled["gap_size"].max() > 0

def test_gap_fill_xyz():

    # Load the data
    xyz_all_base_path = Path(recording_directory,tracker_enum.name, f"xyz_{tracker_enum.name}.csv")
    logger.info(f"Reading in raw xy data located at {xyz_all_base_path}")
    xyz_all_base = pd.read_csv(xyz_all_base_path)
    base_length = xyz_all_base.shape[0]
    xyz_all_filled = gap_fill_xyz(xyz_all_base)
    filled_length = xyz_all_filled.shape[0]

    xyz_filled_path = Path(xyz_all_base_path.parent, f"xyz_{tracker_enum.name}_filled.csv")
    xyz_all_filled.to_csv(xyz_filled_path, index=False)

    # a coupLe basic assertions
    assert base_length < filled_length
    assert xyz_all_filled["gap_size"].max() > 0

if __name__ == "__main__":
    test_gap_fill_xy()
    test_gap_fill_xyz()

