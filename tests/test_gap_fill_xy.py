
#%%
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)


import pandas as pd
import numpy as np

from pathlib import Path
from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.post_processing.gap_filling import xy_gap_fill
from pyxy3d import __root__

def test_gap_fill_xy():
    recording_directory = Path(__root__, "tests", "reference", "base_data")
    tracker_enum = TrackerEnum.HOLISTIC_OPENSIM

    # Load the data
    xy_all_base_path = Path(recording_directory,tracker_enum.name, f"xy_{tracker_enum.name}.csv")
    logger.info(f"Reading in raw xy data located at {xy_all_base_path}")
    xy_all_base = pd.read_csv(xy_all_base_path)
    base_length = xy_all_base.shape[0]
    xy_all_filled = xy_gap_fill(xy_all_base)
    filled_length = xy_all_filled.shape[0]

    assert base_length < filled_length
    assert xy_all_filled["gap_size"].max() > 0

if __name__ == "__main__":
    test_gap_fill_xy()

# Write the DataFrame to a new CSV file
# xy_filled_path = Path(xy_all_base_path.parent, f"xy_{tracker_enum.name}_filled.csv")
# xy_all_filled.to_csv(xy_filled_path, index=False)
# %%
