
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

from pyxy3d.gui.camera_config.camera_config_dialogue import CameraConfigTab
from pyxy3d.session.session import Session, SessionMode
from pyxy3d.session.get_stage import get_camera_stage, CameraStage
from pyxy3d.gui.navigation_bars import NavigationBarBackNext

class CameraWizard(QWidget):
    """ 
    This is basically just the camera tabs plus the navigation bar
    """
    def __init__(self, session: Session):
        super(CameraWizard, self).__init__()
        self.setLayout(QVBoxLayout())    
        self.camera_tabs = CameraTabs(session)
        self.navigation_bar = NavigationBarBackNext()
        self.layout().addWidget(self.camera_tabs)
        self.layout().addWidget(self.navigation_bar)
    
        self.camera_tabs.stereoframe_ready.connect(self.set_next_enabled)
        self.camera_tabs.check_session_calibration()
        self.session = session
        
        #prior to entering intrinisc calibration mode, need to have an active monocalibrator
        # self.session.active_monocalibrator = self.camera_tabs.currentWidget().port
        # self.session.set_mode(SessionMode.Charuco)
         
    def set_next_enabled(self, stereoframe_ready:bool):
        logger.info(f"Setting camera tab next button enabled status to {stereoframe_ready}")
        self.navigation_bar.next_btn.setEnabled(stereoframe_ready)
            
class CameraTabs(QTabWidget):
    
    stereoframe_ready = pyqtSignal(bool)

    def __init__(self, session: Session):
        super(CameraTabs, self).__init__()
        self.session = session

        self.setTabPosition(QTabWidget.TabPosition.North)
        self.add_cam_tabs()
        # self.session.set_mode(SessionMode.IntrinsicCalibration)
        self.currentChanged.connect(self.activate_current_tab)

    def keyPressEvent(self, event):
        """
        Override the keyPressEvent method to allow navigation via PgUp/PgDown
        """

        if event.key() == Qt.Key.Key_PageUp:
            current_index = self.currentIndex()
            if current_index > 0:
                self.setCurrentIndex(current_index - 1)
        elif event.key() == Qt.Key.Key_PageDown:
            current_index = self.currentIndex()
            if current_index < self.count() - 1:
                self.setCurrentIndex(current_index + 1)
        else:
            super().keyPressEvent(event)
        
        
    def activate_current_tab(self, index):

        logger.info(f"Toggle tracking to activate {self.tabText(index)}")
        self.session.pause_all_monocalibrators()
        self.session.activate_monocalibrator(self.widget(index).stream.port)

        # this is where you can update the spin boxes to align with the session values
        monocal_fps = self.session.get_active_mode_fps()
        self.widget(index).advanced_controls.frame_rate_spin.setValue(monocal_fps)

        wait_time_intrinsic = self.session.wait_time_intrinsic
        self.widget(index).advanced_controls.wait_time_spin.setValue(wait_time_intrinsic)

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
                    cam_tab = CameraConfigTab(self.session, port)
                    
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
        
        # self.toggle_tracking(self.currentIndex())

    def check_session_calibration(self):
        logger.info(f"Checking session stage....")
        current_stage = get_camera_stage(self.session) 
        if current_stage == CameraStage.INTRINSICS_ESTIMATED:
            logger.info("Ready for stereoframe")
            self.stereoframe_ready.emit(True)       
        elif current_stage == CameraStage.UNCALIBRATED_CAMERAS:
            logger.info("Not ready for stereoframe")
            self.stereoframe_ready.emit(False)
            
if __name__ == "__main__":
    from pyxy3d import __root__
    
    App = QApplication(sys.argv)

    
    config_path = Path(__root__, "tests", "laptop")
    # config_path = Path(__root__, "tests", "pyxy3d")
    # config_path = Path(repo, "sessions", "high_res_session")
    print(config_path)
    session = Session(config_path)
    # session.load_cameras()
    session.load_stream_tools()

    test_port = 0

    # cam_dialog = CameraConfigDialog(session, test_port)
    # cam_tabs = CameraTabs(session)
    # cam_tabs.show()
    cam_wizard = CameraWizard(session)
    cam_wizard.show()

    sys.exit(App.exec())

