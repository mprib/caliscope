
from PyQt6.QtWidgets import QApplication

from pyxy3d import __root__
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from pathlib import Path
import sys
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
import pickle
from pyxy3d.session import Session
from pyxy3d.configurator import Configurator
from pyxy3d.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget

session_path = Path(__root__, "tests" , "sessions_copy_delete", "2_cam_set_origin_test")
config = Configurator(session_path)
session = Session(config)

session.load_estimated_capture_volume()

app = QApplication(sys.argv)

vizr_dialog = CaptureVolumeWidget(session)
vizr_dialog.show()

sys.exit(app.exec())