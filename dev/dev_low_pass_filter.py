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

import matplotlib.pyplot as plt
# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents
from pyxy3d.post_processor import PostProcessor
from pyxy3d.trackers.tracker_enum import TrackerEnum

#%%


# load config:
test_folder = Path(__root__, r"tests\reference\2d_data")
config = Configurator(test_folder)
logger.info("Loading data...")
xyz_history = pd.read_csv(Path(test_folder, "xyz_HOLISTIC.csv"))

#%%
# for initial testing, just pull out the left heel
# in the future, may rearrange this to just start from xy and filter 
# left heel then go from there...
left_heel_id = 29
xyz_history = xyz_history.query(f"point_id=={left_heel_id}")
#%%



# load the data
# test_data_path = Path(__root__, r"tests\reference\2d_data\xy_HOLISTIC.csv")
# xy_data = pd.read_csv(test_data_path)
# Create a post processor
# post_processor = PostProcessor(config)
# xyz_history = post_processor.triangulate_xy_data(xy_data)
# xyz_history = pd.DataFrame(xyz_history)
# xyz_history.to_csv(Path(test_folder, "xyz_HOLISTIC.csv"))
#%%
# creating filter groups which are contiguous observations of a single point
xyz_history = xyz_history.sort_values(by=["point_id", "sync_index"])
xyz_history["sync_index_shifted"] = xyz_history["sync_index"].shift(1)
xyz_history["new_filter_group"] = xyz_history["sync_index"] != xyz_history["sync_index_shifted"] + 1
xyz_history["filter_group_index"] = xyz_history["new_filter_group"].cumsum()
xyz_history = xyz_history.drop(["sync_index_shifted", "new_filter_group"], axis=1)

#%%
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
    return y

#%%
# Define your filter parameters
order = 2

fs = config.get_fps_recording()  # sample rate, Hz

cutoff = 3 # desired cutoff frequency, Hz

logger.info("Applying butterworth filter to xy point coordinates")
# Apply the filter to each piecewise group
xyz_history["filtered_x_coord"] = xyz_history.groupby(["filter_group_index"])[
    "x_coord"
].transform(butter_lowpass_filter, cutoff, fs, order)
xyz_history["filtered_y_coord"] = xyz_history.groupby(["filter_group_index"])[
    "y_coord"
].transform(butter_lowpass_filter, cutoff, fs, order)
xyz_history["filtered_z_coord"] = xyz_history.groupby(["filter_group_index"])[
    "z_coord"
].transform(butter_lowpass_filter, cutoff, fs, order)

xyz_history = xyz_history.sort_values(["sync_index", "point_id"])

# Creating subplots
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plotting x-coordinates
ax[0].plot(xyz_history['sync_index'], xyz_history['x_coord'], color='blue', label='Original')
ax[0].plot(xyz_history['sync_index'], xyz_history['filtered_x_coord'], color='red', label='Filtered')
ax[0].set_title('X Coordinates')
ax[0].set_xlabel('sync_index')
ax[0].set_ylabel('x_coord')
ax[0].legend()

# Plotting y-coordinates
ax[1].plot(xyz_history['sync_index'], xyz_history['y_coord'], color='green', label='Original')
ax[1].plot(xyz_history['sync_index'], xyz_history['filtered_y_coord'], color='purple', label='Filtered')
ax[1].set_title('Y Coordinates')
ax[1].set_xlabel('sync_index')
ax[1].set_ylabel('y_coord')
ax[1].legend()

# Plotting z-coordinates
ax[2].plot(xyz_history['sync_index'], xyz_history['z_coord'], color='orange', label='Original')
ax[2].plot(xyz_history['sync_index'], xyz_history['filtered_z_coord'], color='brown', label='Filtered')
ax[2].set_title('Z Coordinates')
ax[2].set_xlabel('sync_index')
ax[2].set_ylabel('z_coord')
ax[2].legend()

# Adjusting the spacing between the plots
plt.tight_layout()

# Displaying the plot
plt.show()
# %%
