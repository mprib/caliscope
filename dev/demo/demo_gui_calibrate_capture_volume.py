from PyQt6.QtWidgets import QApplication
from pyxy3d.gui.calibrate_capture_volume_widget import CalibrateCaptureVolumeWidget
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

config = Configurator(session_path)
session = Session(config)
session.set_mode(SessionMode.ExtrinsicCalibration)
app = QApplication(sys.argv)
window = CalibrateCaptureVolumeWidget(session)

window.show()

app.exec()