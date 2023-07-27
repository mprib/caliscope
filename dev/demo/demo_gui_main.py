from PySide6.QtWidgets import QApplication
import sys
from pyxy3d import __root__
from pathlib import Path
import toml
from pyxy3d import __app_dir__
from pyxy3d.gui.single_main_widget import MainWindow

app_settings = toml.load(Path(__app_dir__, "settings.toml"))
recent_projects: list = app_settings["recent_projects"]

recent_project_count = len(recent_projects)
session_path = Path(recent_projects[recent_project_count - 1])

app = QApplication(sys.argv)
window = MainWindow()
window.launch_session(str(session_path))
window.show()
app.exec()
