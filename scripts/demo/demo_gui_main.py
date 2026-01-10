from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.gui.main_widget import MainWindow

try:
    import qdarktheme  # type: ignore[import-not-found]
except ImportError:
    qdarktheme = None

app = QApplication(sys.argv)
if qdarktheme is not None:
    qdarktheme.setup_theme("auto")

window = MainWindow()
workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\caliscope\4_cam_prerecorded_practice_working")

window.launch_workspace(str(workspace_dir))
window.show()
app.exec()
