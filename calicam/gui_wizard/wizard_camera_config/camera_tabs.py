
import calicam.logger
logger = calicam.logger.get(__name__)

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QTabWidget,
)

from calicam.gui_wizard.wizard_camera_config.camera_config_dialogue import CameraConfigDialog
from calicam.session import Session

class CameraTabs(QTabWidget):
    
    def __init__(self, session):
        super(CameraTabs, self).__init__()
        self.session = session

        self.setTabPosition(QTabWidget.TabPosition.North)
        self.add_cam_tabs()

    def add_cam_tabs(self):
        tab_names = [self.tabText(i) for i in range(self.count())]
        logger.info(f"Current tabs are: {tab_names}")

        if len(self.session.streams) > 0:
            for port, stream in self.session.streams.items():
                tab_name = f"Camera {port}"
                
                logger.info(f"Potentially adding {tab_name}")
                if tab_name in tab_names:
                    pass  # already here, don't bother
                else:
                    cam_tab = CameraConfigDialog(self.session, port)

                    self.insertTab(port, cam_tab, tab_name)
        else:
            logger.info("No cameras available")


if __name__ == "__main__":
    App = QApplication(sys.argv)

    repo = Path(str(Path(__file__)).split("calicam")[0],"calicam")
    # config_path = Path(repo, "sessions", "high_res_session")
    config_path = Path(repo, "sessions", "5_cameras")
    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    # session.adjust_resolutions()
    # session.load_monocalibrators()

    test_port = 0

    # cam_dialog = CameraConfigDialog(session, test_port)
    cam_tabs = CameraTabs(session)
    
    cam_tabs.show()

    sys.exit(App.exec())

