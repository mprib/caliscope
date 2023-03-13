
import pyxyfy.logger
logger = pyxyfy.logger.get(__name__)

import sys
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

from pyxyfy.gui.camera_config.camera_config_dialogue import CameraConfigDialog
from pyxyfy.session import Session, Stage
from pyxyfy.gui.widgets import NavigationBarBackNext

class CameraWizard(QWidget):
    def __init__(self, session):
        super(CameraWizard, self).__init__()
        self.setLayout(QVBoxLayout())    
        self.camera_tabs = CameraTabs(session)
        self.navigation_bar = NavigationBarBackNext()
        self.layout().addWidget(self.camera_tabs)
        self.layout().addWidget(self.navigation_bar)
    
        self.camera_tabs.omniframe_ready.connect(self.set_next_enabled)
        self.camera_tabs.check_session_calibration()
         
    def set_next_enabled(self, omniframe_ready:bool):
        logger.info(f"Setting camera tab next button enabled status to {omniframe_ready}")
        self.navigation_bar.next_btn.setEnabled(omniframe_ready)
            
class CameraTabs(QTabWidget):
    
    omniframe_ready = pyqtSignal(bool)

    def __init__(self, session: Session):
        super(CameraTabs, self).__init__()
        self.session = session

        self.setTabPosition(QTabWidget.TabPosition.North)
        self.add_cam_tabs()
        self.currentChanged.connect(self.toggle_tracking)
        
        
    def toggle_tracking(self, index):

        logger.info(f"Toggle tracking to activate {self.tabText(index)}")
        self.session.set_active_monocalibrator(self.widget(index).stream.port)


    def add_cam_tabs(self):
        tab_names = [self.tabText(i) for i in range(self.count())]
        logger.info(f"Current tabs are: {tab_names}")

        if len(self.session.monocalibrators) > 0:
            
            # construct a dict of tabs so that they can then be placed in order
            tab_widgets = {}
            for port, monocal in self.session.monocalibrators.items():
                tab_name = f"Camera {port}"
                logger.info(f"Potentially adding {tab_name}")
                if tab_name in tab_names:
                    pass  # already here, don't bother
                else:
                    cam_tab = CameraConfigDialog(self.session, port)
                    
                    # when new camera calibrated, check to see if all cameras calibrated
                    cam_tab.calibrate_grp.calibration_change.connect(self.check_session_calibration)

                    tab_widgets[port] = cam_tab

            # add the widgets to the tab bar in order
            ordered_ports = list(tab_widgets.keys())
            ordered_ports.sort()
            for port in ordered_ports:
                self.insertTab(port, tab_widgets[port], f"Camera {port}")
            
            # session may be pre-calibrated and ready to proceed...or not
            self.check_session_calibration()
            
        else:
            logger.info("No cameras available")
        
        self.toggle_tracking(self.currentIndex())
    
    def check_session_calibration(self):
        logger.info(f"Checking session stage....")
        if self.session.get_stage() == Stage.MONOCALIBRATED_CAMERAS:
            logger.info("Ready for omniframe")
            self.omniframe_ready.emit(True)       
        elif self.session.get_stage() == Stage.UNCALIBRATED_CAMERAS:
            logger.info("Not ready for omniframe")
            self.omniframe_ready.emit(False)
            
if __name__ == "__main__":
    from pyxyfy import __root__
    
    App = QApplication(sys.argv)

    
    # config_path = Path(__root__, "sessions", "laptop")
    config_path = Path(__root__, "tests", "pyxyfy")
    # config_path = Path(repo, "sessions", "high_res_session")
    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    # session.adjust_resolutions()
    session.load_monocalibrators()

    test_port = 0

    # cam_dialog = CameraConfigDialog(session, test_port)
    # cam_tabs = CameraTabs(session)
    # cam_tabs.show()
    cam_wizard = CameraWizard(session)
    cam_wizard.show()

    sys.exit(App.exec())

