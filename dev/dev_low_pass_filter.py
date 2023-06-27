"""
This may be working somewhat now... looking at the 3d reconstructions, it doesn't seem to make much 
of an impact on the final results, and it takes a considerable amount of time to process. 
I imagine that much of this is from the face data. but it's still more than I think it is worth. 
I'll push this back to main to have this here for future reference, but otherwise I'm going to 
once again table the idea of using filtering of some kind.
"""
# %%

import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
from pyxy3d.configurator import Configurator
from pathlib import Path
from pyxy3d import __root__
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents
from pyxy3d.post_processor import PostProcessor
from pyxy3d.trackers.tracker_enum import TrackerEnum


# Define your Butterworth filter functions
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)

    # need to adjust for short input sequences
    padlen = min(len(data) - 1, 3 * (max(len(a), len(b)) - 1))
    y = filtfilt(b, a, data, padlen=padlen)
    y = np.round(y, 1)
    return y


# load config:
test_folder = Path(__root__, r"tests\reference\2d_data")
config = Configurator(test_folder)

logger.info("Loading data...")
# load the data
test_data_path = Path(__root__, r"tests\reference\2d_data\xy_HOLISTIC.csv")
# %%
data = pd.read_csv(test_data_path)
data = data.sort_values(by=["port", "point_id", "frame_index"])
data["frame_index_shifted"] = data["frame_index"].shift(1)
data["new_filter_group"] = data["frame_index"] != data["frame_index_shifted"] + 1
data["filter_group_index"] = data["new_filter_group"].cumsum()
data = data.drop(["frame_index_shifted", "new_filter_group"], axis=1)


# Define your filter parameters
order = 2
fs = config.get_fps_recording()  # sample rate, Hz
cutoff = 6.0  # desired cutoff frequency, Hz
# %%

logger.info("Applying butterworth filter to xy point coordinates")
# Apply the filter to each piecewise group
data["filtered_img_loc_x"] = data.groupby(["filter_group_index"])[
    "img_loc_x"
].transform(butter_lowpass_filter, cutoff, fs, order)
data["filtered_img_loc_y"] = data.groupby(["filter_group_index"])[
    "img_loc_y"
].transform(butter_lowpass_filter, cutoff, fs, order)

# %%
data = data.sort_values(["sync_index", "point_id"])
logger.info("Saving out data...")
# %%
data.to_csv(Path(test_folder, "HOLISTIC_filtered_xy.csv"))
# %%
