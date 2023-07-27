
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

from PySide6.QtWidgets import (
    QApplication,
)
import toml

from pyxy3d.gui.recording_widget import RecordingWidget
from pyxy3d.session.session import Session, SessionMode
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.helper import copy_contents
from pyxy3d.configurator import Configurator
from pyxy3d import __app_dir__

app_settings = toml.load(Path(__app_dir__, "settings.toml"))
recent_projects:list = app_settings["recent_projects"]

recent_project_count = len(recent_projects)
session_path = Path(recent_projects[recent_project_count-1])
# copy_contents(session_origin_path,session_path)
config = Configurator(session_path)
session = Session(config)
session.set_mode(SessionMode.Recording)
# while not hasattr(session.synchronizer, "current_sync_packet"):
#     logger.info("Waiting for session synchronizer to load")
#     time.sleep(1)

App = QApplication(sys.argv)
# App = QApplication([])
logger.info("About to create Recording Widget")
recording_dialog = RecordingWidget(session)


logger.info("recording dialog assigned")
recording_dialog.show()
logger.info("recording dialog shown")
sys.exit(App.exec())