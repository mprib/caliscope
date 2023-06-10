
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from PyQt6.QtWidgets import QApplication
from pyxy3d.gui.camera_config.intrinsic_calibration_widget import IntrinsicCalibrationWidget
import sys
from time import sleep
from pyxy3d import __root__
from pathlib import Path
from pyxy3d.configurator import Configurator
from pyxy3d.session.session import Session, SessionMode
import toml
from pyxy3d import __app_dir__

app_settings = toml.load(Path(__app_dir__, "settings.toml"))
recent_projects:list = app_settings["recent_projects"]

recent_project_count = len(recent_projects)
session_path = Path(recent_projects[recent_project_count-1])
logger.info(f"Attempting to launch session stored at {session_path}")

config = Configurator(session_path)
session = Session(config)
session.connect_to_cameras()
session.set_mode(SessionMode.IntrinsicCalibration) 
app = QApplication(sys.argv)
window = IntrinsicCalibrationWidget(session)


window.show()

app.exec()