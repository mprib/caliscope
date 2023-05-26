import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

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
from pyxy3d.session.session import Session
from pyxy3d.configurator import Configurator
from pyxy3d.gui.calibration_widget import CalibrationWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.app_settings = toml.load(__settings_path__)

        self.setWindowTitle("PyXY3D")
        self.setWindowIcon(QIcon(str(Path(__root__, "pyxy3d/gui/icons/pyxy_logo.svg"))))
        self.setMinimumSize(500, 500)

        # File Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&File")

        # Open or New project (can just create a folder in the dialog in truly new)
        self.open_project_action = QAction("&New/Open Project", self)
        self.open_project_action.triggered.connect(self.create_new_project_folder)
        self.file_menu.addAction(self.open_project_action)


        # Open Recent
        self.open_recent_project_submenu = QMenu("&Recent Projects...", self)
        # Populate the submenu with recent project paths
        for project_path in self.app_settings["recent_projects"]:
            self.add_to_recent_project(project_path)

        self.file_menu.addMenu(self.open_recent_project_submenu)

        self.close_session_action = QAction("&Close Session", self)        
        self.close_session_action.triggered.connect(self.close_current_session)
        self.file_menu.addAction(self.close_session_action)


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

    def close_current_session(self):
        pass


    def launch_session(self, path_to_folder:str):

        session_path = Path(path_to_folder)
        self.config = Configurator(session_path)
        logger.info(f"Launching session with config file stored in {session_path}")
        self.session = Session(self.config)
        logger.info("Setting calibration Widget")

        # if calibration widget is currently selected, make sure it stays selected
        old_index = self.tab_widget.currentIndex()
        calibration_index = self.tab_widget.indexOf(self.calibration_widget)
        self.tab_widget.removeTab(calibration_index)
        self.calibration_widget.deleteLater()
        new_calibration_widget = CalibrationWidget(self.session)
        # self.tab_widget.setTabText(calibration_index, "&Calibration")  # Set the tab text
        self.tab_widget.insertTab(calibration_index, new_calibration_widget, "&Calibration")
        self.calibration_widget = new_calibration_widget
        self.tab_widget.setCurrentIndex(old_index)

        
    def add_to_recent_project(self, project_path:str):
        recent_project_action = QAction(project_path, self)
        recent_project_action.triggered.connect(self.open_recent_project)
        self.open_recent_project_submenu.addAction(recent_project_action)

    def open_recent_project(self):
        action = self.sender()
        project_path = action.text()
        logger.info(f"Opening recent session stored at {project_path}")
        self.launch_session(project_path)




    def create_new_project_folder(self):
        default_folder = Path(self.app_settings["last_project_parent"])
        dialog = QFileDialog()
        path_to_folder = dialog.getExistingDirectory(
            parent=None,
            caption="Open Previous or Create New Project Directory",
            directory=str(default_folder),
            options=QFileDialog.Option.ShowDirsOnly,
        )
        
        if path_to_folder:
            logger.info(("Creating new project in :", path_to_folder))
            self.add_project_to_recent(path_to_folder)
            self.launch_session(path_to_folder)
            
    
    def add_project_to_recent(self, folder_path):
        if str(folder_path) in self.app_settings["recent_projects"]:
            pass
        else:
            self.app_settings["recent_projects"].append(str(folder_path))
            self.app_settings["last_project_parent"] = str(Path(folder_path).parent)
            self.update_app_settings()
            self.add_to_recent_project(folder_path)

    def update_app_settings(self):
        with open(__settings_path__, "w") as f:
            toml.dump(self.app_settings, f)

def launch_main():
    from pyxy3d.gui.qt_logger import QtLogger
    app = QApplication([])
    log_widget = QtLogger()
    log_widget.show()
    window = MainWindow()
    window.show()
    
    app.exec()

if __name__ == "__main__":
    launch_main()