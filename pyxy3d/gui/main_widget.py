import pyxy3d.logger

from PyQt6.QtWidgets import QMainWindow, QStackedLayout, QFileDialog

logger = pyxy3d.logger.get(__name__)
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStackedLayout,
    QWidget,
    QVBoxLayout,
    QMenu,
    QMenuBar,
    QTabWidget,
)
import toml
from PyQt6.QtGui import QIcon, QAction, QKeySequence, QShortcut
from PyQt6.QtCore import Qt
from pyxy3d import __root__, __settings_path__, __user_dir__


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = toml.load(__settings_path__)

        self.setWindowTitle("PyXY3D")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxy3d/gui/icons/pyxy_logo.svg"))))
        self.setMinimumSize(500, 500)

        # Set up menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&File")
        self.new_project_action = QAction("&New Project", self)
        self.open_project_action = QAction("&Open Project", self)
        self.file_menu.addAction(self.new_project_action)
        self.file_menu.addAction(self.open_project_action)
        self.open_recent_project = QMenu("Open &Recent Project", self)
        self.file_menu.addMenu(self.open_recent_project)

        self.new_project_action

        self.cameras_menu = self.menu.addMenu("Ca&meras")
        self.disconnect_cameras_action = QAction("&Disconnect Cameras", self)
        self.connect_cameras_action = QAction("Co&nnect Cameras", self)
        self.cameras_menu.addAction(self.disconnect_cameras_action)
        self.cameras_menu.addAction(self.connect_cameras_action)

        # Set up tabs
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.calibration_widget = QWidget()
        self.recording_widget = QWidget()
        self.processing_widget = QWidget()

        self.tab_widget.addTab(self.calibration_widget, "&Calibration")
        self.tab_widget.addTab(self.recording_widget, "&Recording")
        self.tab_widget.addTab(self.processing_widget, "&Processing")

        self.connect_actions()

    def connect_actions(self):
        self.new_project_action.triggered.connect(self.create_new_project_folder)

    def create_new_project_folder(self):
        default_folder = Path(self.app_settings["last_project_parent"])
        dialog = QFileDialog()
        # dialog.setFileMode(QFileDialog.Option.ShowDirsOnly)
        # dialog.setViewMode()
        folder_path = dialog.getExistingDirectory(
            parent=None,
            caption="Create Project Directory",
            directory=str(default_folder),
            options=QFileDialog.Option.ShowDirsOnly,
        )
        
        if folder_path:
            logger.info(("Creating new project in :", folder_path))
            self.add_project_to_recent(folder_path)
            
    
    def add_project_to_recent(self, folder_path):
        if str(folder_path) in self.app_settings["recent_projects"]:
            pass
        else:
            self.app_settings["recent_projects"].append(str(folder_path))
            self.app_settings["last_project_parent"] = str(Path(folder_path).parent)
            self.update_app_settings()

    def update_app_settings(self):
        with open(__settings_path__, "w") as f:
            toml.dump(self.app_settings, f)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
