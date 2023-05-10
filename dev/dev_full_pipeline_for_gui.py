import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
from pyxy3d.configurator import Configurator
from pathlib import Path
from pyxy3d import __root__
import pandas as pd
from pyxy3d.trackers.tracker_enum import Tracker as trackers
# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents
from pyxy3d.post_processing_pipelines import create_xyz


session_path = Path(__root__, "tests", "sessions", "4_cam_recording")
copy_session_path = Path(__root__, "tests", "sessions_copy_delete", "4_cam_recording")

copy_contents(session_path, copy_session_path)

# create inputs to processing pipeline function
config = Configurator(copy_session_path)

recording_directory = Path(copy_session_path, "recording_1")

create_xyz(
    session_path=config.session_path,
    recording_path=recording_directory,
    tracker_enum=trackers.POSE
)
