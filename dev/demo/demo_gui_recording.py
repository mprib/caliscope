
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
import math
from pathlib import Path
from threading import Thread, Event
import numpy as np
import time
from queue import Queue
import shutil

from PyQt6.QtWidgets import (
    QApplication,
)

from pyxy3d.gui.recording_widget import RecordingWidget
from pyxy3d.session.session import Session
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.helper import copy_contents
from pyxy3d.configurator import Configurator

# session_origin_path = Path(__root__, "dev", "sample_sessions", "low_res")
# session_origin_path = Path(__root__, "dev", "sample_sessions", "low_res_laptop")
# session_origin_path = Path(__root__, "dev","sample_sessions", "recordings_to_process")
# session_path = Path(__root__, "dev","sample_sessions", "296")
session_path = Path(__root__, "tests" , "sessions_copy_delete", "2_cam_set_origin_test")

# clear previous test so as not to pollute current test results
# if session_path.exists() and session_path.is_dir():
    # shutil.rmtree(session_path)   

# copy_contents(session_origin_path,session_path)
config = Configurator(session_path)
session = Session(config)
# session.load_cameras()
session.load_streams()
session.adjust_resolutions()


App = QApplication(sys.argv)
recording_dialog = RecordingWidget(session)
recording_dialog.show()

sys.exit(App.exec())