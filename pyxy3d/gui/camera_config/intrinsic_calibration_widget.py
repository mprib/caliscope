
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
from pyxy3d.session.session import Session
from pyxy3d.gui.navigation_bars import NavigationBarBackNext

class IntrinsicCalibrationWidget(QWidget):
    """ 
    This is basically just the camera tabs plus the navigation bar
    """
    def __init__(self, session: Session):
        super(IntrinsicCalibrationWidget, self).__init__()
        self.setLayout(QVBoxLayout())    
        self.camera_tabs = CameraTabs(session)
        self.layout().addWidget(self.camera_tabs)
        self.session = session
        
            
class CameraTabs(QTabWidget):
    
    stereoframe_ready = pyqtSignal(bool)

    def __init__(self, session: Session):
        super(CameraTabs, self).__init__()
        self.session = session

        self.setTabPosition(QTabWidget.TabPosition.North)
        self.add_cam_tabs()
        # self.session.set_mode(SessionMode.IntrinsicCalibration)
        self.currentChanged.connect(self.activate_current_tab)
        self.session.activate_monocalibrator(self.currentWidget().stream.port)

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
                    tab_widgets[port] = cam_tab

            # add the widgets to the tab bar in order
            ordered_ports = list(tab_widgets.keys())
            ordered_ports.sort()
            for port in ordered_ports:
                self.insertTab(port, tab_widgets[port], f"Camera {port}")
            
        else:
            logger.info("No cameras available")
        