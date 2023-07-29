
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
from pyxy3d.post_processing.smoothing import _smooth_xy, smooth_xyz
from pyxy3d.post_processing.gap_filling import gap_fill_xy, gap_fill_xyz
from pyxy3d import __root__

original_base_data_directory = Path(__root__, "tests", "reference", "base_data")
base_data_directory = Path(original_base_data_directory.parent.parent, "reference_delete", "base_data")
copy_contents(original_base_data_directory, base_data_directory)

xyz_path = Path(base_data_directory, "HOLISTIC_OPENSIM", "xyz_HOLISTIC_OPENSIM.csv")
xy_path = Path(base_data_directory,"HOLISTIC_OPENSIM", "xy_HOLISTIC_OPENSIM.csv")

config = Configurator(base_data_directory)

config_path = Path(base_data_directory, "config.toml")
config_dict = toml.load(config_path)

# Define your filter parameters
order = 2
fs = config.get_fps_recording()  # sample rate, Hz

# note that the cutoff must be < 0.5*(sampling rate, a.k.a. nyquist frequency)
cutoff = 6  # desired cutoff frequency, Hz

logger.info("Loading data...")
xy  = pd.read_csv(xy_path)
xy = gap_fill_xy(xy)

xyz = pd.read_csv(xyz_path)
xyz = gap_fill_xyz(xyz)
xyz_filtered = smooth_xyz(xyz, order,fs, cutoff)

# save out the filterd data
destination_path = Path(xyz_path.parent, xyz_path.stem + "_filtered.csv")
logger.info(f"Saving filtered data to {destination_path}")
xyz_filtered.to_csv(destination_path)

app = QApplication(sys.argv)

# load in the data for the playback
camera_array = config.get_camera_array()
filtered_data_path =Path(base_data_directory, "HOLISTIC_OPENSIM", "xyz_HOLISTIC_OPENSIM_filtered.csv") 
filtered_data = pd.read_csv(filtered_data_path)

# create and show the playback widget
filtered_window = PlaybackTriangulationWidget(camera_array)
filtered_window.set_xyz(filtered_data)
filtered_window.show()

app.exec()
