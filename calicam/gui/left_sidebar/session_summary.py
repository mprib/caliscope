
import calicam.logger
logger = calicam.logger.get(__name__)
import sys

from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calicam.session import Session
from calicam.gui.left_sidebar.charuco_summary import CharucoSummary
from calicam.gui.left_sidebar.camera_summary import CameraSummary
from calicam.gui.left_sidebar.fps_control import FPSControl


class SessionSummary(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session

        vbox = QVBoxLayout()
        self.setLayout(vbox)
        
        folder_grp = QGroupBox("Session")
        folder_grp.setLayout(QVBoxLayout())

        folder_grp.layout().addWidget(QLabel(f"Name: {session.folder}"))
        self.stage_label = QLabel(f"Stage: {session.get_stage()}")
        folder_grp.layout().addWidget(self.stage_label)
        vbox.addWidget(folder_grp)


        charuco_grp = QGroupBox("Charuco Board")
        charuco_grp.setLayout(QHBoxLayout())
        self.charuco_summary = CharucoSummary(self.session)
        charuco_grp.layout().addWidget(self.charuco_summary)
        vbox.addWidget(charuco_grp)
       

       
        cam_grp = QGroupBox("Saved Cameras") 
        cam_grp.setLayout(QHBoxLayout())
        self.camera_summary = CameraSummary(self.session)
        cam_grp.layout().addWidget(self.camera_summary)
        vbox.addWidget(cam_grp)



if __name__ == "__main__":
    repo = Path(str(Path(__file__)).split("calicam")[0],"calicam").parent
    config_path = Path(repo, "sessions", "high_res_session")
    
    session = Session(config_path)
    print(session.config)
    app = QApplication(sys.argv)

    side_bar = SessionSummary(session)
    side_bar.show()

    sys.exit(app.exec())