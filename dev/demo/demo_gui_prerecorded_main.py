from PySide6.QtWidgets import QApplication
import sys
from pyxy3d import __root__
from pathlib import Path
import toml
from pyxy3d import __app_dir__
from pyxy3d.gui.prerecorded_main_widget import PreRecordedMainWindow
from PySide6.QtWidgets import QApplication
import qdarktheme

app = QApplication(sys.argv)
qdarktheme.setup_theme("auto")

window = PreRecordedMainWindow()
workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\4_cam_prerecorded_practice_working")

window.launch_workspace(str(workspace_dir))
window.controller
window.controller.load_camera_array()
window.controller.load_intrinsic_stream_manager()
window.show()
app.exec()