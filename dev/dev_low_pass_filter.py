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
from pyxy3d.post_processing.post_processor import PostProcessor
from pyxy3d.trackers.tracker_enum import TrackerEnum
import toml
from PySide6.QtWidgets import QApplication
from pyxy3d.gui.vizualize.playback_triangulation_widget import PlaybackTriangulationWidget


# %%
# load config:



# %%
# for initial testing, just pull out the left heel
# in the future, may rearrange this to just start from xy and filter
# left heel then go from there...
# left_heel_id = 29
# xyz_history = xyz_history.query(f"point_id=={left_heel_id}")
# %%

# load the data
# test_data_path = Path(__root__, r"tests\reference\2d_data\xy_HOLISTIC.csv")
# xy_data = pd.read_csv(test_data_path)
# Create a post processor
# post_processor = PostProcessor(config)
# xyz_history = post_processor.triangulate_xy_data(xy_data)
# xyz_history = pd.DataFrame(xyz_history)
# xyz_history.to_csv(Path(test_folder, "xyz_HOLISTIC.csv"))
# creating filter groups which are contiguous observations of a single point


# %%
# Define Butterworth filter functions
def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)

    # need to adjust for short input sequences
    padlen = min(len(data) - 1, 3 * (max(len(a), len(b)) - 1))
    y = filtfilt(b, a, data, padlen=padlen)
    return y


def filter_xyz(xyz: pd.DataFrame, order, fs, cutoff)->pd.DataFrame:
    xyz = xyz.sort_values(by=["point_id", "sync_index"])
    xyz["sync_index_shifted"] = xyz["sync_index"].shift(1)
    xyz["new_filter_group"] = (
        xyz["sync_index"] != xyz["sync_index_shifted"] + 1
    )
    xyz["filter_group_index"] = xyz["new_filter_group"].cumsum()
    xyz = xyz.drop(["sync_index_shifted", "new_filter_group"], axis=1)

    logger.info("Applying butterworth filter to xy point coordinates")
    # Apply the filter to each piecewise group

    for coord in ["x_coord", "y_coord", "z_coord"]:
        xyz[coord] = xyz.groupby(["filter_group_index"])[
            coord
        ].transform(butter_lowpass_filter, cutoff, fs, order)

    
    xyz = xyz.sort_values(["sync_index", "point_id"])
    
    return xyz   


processed_folder = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\4_cam_A\recording_4\HOLISTIC_OPENSIM")
xyz_path = Path(processed_folder, "xyz_HOLISTIC_OPENSIM.csv")

config = Configurator(processed_folder)

config_path = Path(processed_folder, "config.toml")
config_dict = toml.load(config_path)

fps = config_dict["fps_recording"]

# Define your filter parameters
order = 2
fs = config.get_fps_recording()  # sample rate, Hz
# note that the cutoff must be < 0.5*(sampling rate, a.k.a. nyquist frequency)
cutoff = 6  # desired cutoff frequency, Hz

logger.info("Loading data...")
xyz = pd.read_csv(xyz_path)

xyz_filtered = filter_xyz(xyz, order,fs, cutoff)

# save out the filterd data
destination_path = Path(xyz_path.parent, xyz_path.stem + "_filtered.csv")
logger.info(f"Saving filtered data to {destination_path}")
xyz_filtered.to_csv(destination_path)

app = QApplication(sys.argv)

# load in the data for the playback
camera_array = config.get_camera_array()
filtered_data_path =Path(processed_folder, "xyz_HOLISTIC_OPENSIM_filtered.csv") 
# filtered_data_path =Path(processed_folder, "xyz_HOLISTIC_OPENSIM.csv") 
filtered_data = pd.read_csv(filtered_data_path)

# create and show the playback widget
filtered_window = PlaybackTriangulationWidget(camera_array)
filtered_window.set_xyz(filtered_data)
filtered_window.show()

app.exec()
