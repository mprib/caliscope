"""
Building test regarding the conversion of the xy.csv datafile into an xyz.csv datafile.
I suppose that it may make sense to run this through the same processing pipeline 
as the SyncPacketTriangulator...no sense reinventing the wheel.



"""

# %%

import pandas as pd
from pathlib import Path

from pyxy3d.helper import copy_contents
from pyxy3d import __root__
from pyxy3d.configurator import Configurator
from pyxy3d.post_processing_pipelines import create_xy_points
from pyxy3d.trackers.hand_tracker import HandTrackerFactory

# load in file of xy point data
origin_data = Path(__root__, "tests", "sessions", "4_cam_recording")
working_data = Path(__root__, "tests", "sessions_copy_delete", "4_cam_recording")

copy_contents(origin_data, working_data)

config = Configurator(working_data)
recording_directory = Path(working_data, "recording_1")
xy_path = Path(recording_directory, "xy.csv")


# need to initially create the xy data....
create_xy_points(
    config=config,
    recording_directory=recording_directory,
    tracker_factory=HandTrackerFactory(),
)

xy_data = pd.read_csv(xy_path)


# %%
