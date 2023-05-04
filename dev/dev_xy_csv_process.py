"""
This may actually end up as a test (please, let that be the case...).
I want to set up a pipeline that could be used by the session to run the 
tracker point detection on each of the video files and save out
the xy point data. This may end up just being incapsulated in a simple
helper function that gets called from the session. I could do some
basic assertions at the end of it to make sure *something* happened.

All I need to do here is run the tracking on an mp4 and save the points out.
This is currenlty happening as part of the calibration process, but it should
also be happening during all of the initial processing.

This xy.csv file will provide the means to rapidly triangulate with 
filtering of the data using reprojection error...

"""



import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)
from time import sleep

import sys
from PyQt6.QtWidgets import QApplication
from pyxy3d.configurator import Configurator
from pathlib import Path
from pyxy3d import __root__
from pyxy3d.trackers.holistic_tracker import HolisticTrackerFactory, HolisticTracker
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.triangulate.real_time_triangulator import SyncPacketTriangulator

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents

session_path = Path(__root__, "dev", "sample_sessions", "293")
copy_session_path = Path(__root__, "dev", "sessions_copy_delete", "293")
copy_contents(session_path, copy_session_path)


config = Configurator(copy_session_path)
camera_array: CameraArray = config.get_camera_array()

# create a tracker
tracker = HolisticTracker()

recording_path = Path(copy_session_path, "recording_1")

# loop through the recording directory
for item in recording_path.iterdir():
    # identify the video files to be processed
    if item.suffix == ".mp4":
        
        logger.info(item)

    





# 