
from PyQt6.QtWidgets import QApplication
import sys
from pyxy3d import __root__
from pathlib import Path
import toml
from pyxy3d import __app_dir__
from pyxy3d.gui.main_widget import MainWindow
from pyxy3d.configurator import Configurator
from pyxy3d.gui.camera_config.intrinsic_calibration_widget import IntrinsicCalibrationWidget
from pyxy3d.session.session import Session, SessionMode


app_settings = toml.load(Path(__app_dir__, "settings.toml"))
recent_projects: list = app_settings["recent_projects"]

recent_project_count = len(recent_projects)
session_path = Path(recent_projects[recent_project_count - 1])
config = Configurator(session_path)
session = Session(config)
session.set_mode(SessionMode.IntrinsicCalibration)

app = QApplication(sys.argv)
window = IntrinsicCalibrationWidget(session)
window.show()

app.exec()