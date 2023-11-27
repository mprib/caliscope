from PySide6.QtWidgets import QApplication
from pyxy3d.gui.calibrate_capture_volume_widget import CalibrateCaptureVolumeWidget
from pyxy3d.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
import sys
from time import sleep
from pyxy3d import __root__
from pathlib import Path
from pyxy3d.configurator import Configurator
from pyxy3d.session.session import LiveSession, SessionMode
import toml
from pyxy3d import __app_dir__
from pyxy3d.controller import Controller

app = QApplication(sys.argv)
workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\4_cam_prerecorded_practice_working")

controller = Controller(workspace_dir)
controller.load_camera_array()
controller.process_extrinsic_streams()
controller.load_intrinsic_streams()

window = CaptureVolumeWidget(controller)

window.show()

app.exec()