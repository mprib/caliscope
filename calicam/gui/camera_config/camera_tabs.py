
import calicam.logger
logger = calicam.logger.get(__name__)

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

from calicam.gui.camera_config.camera_config_dialogue import CameraConfigDialog
from calicam.session import Session
from calicam.gui.widgets import NavigationBarBackNext

class CameraWizard(QWidget):
    def __init__(self, session):
        super(CameraWizard, self).__init__()
        self.setLayout(QVBoxLayout())    
        self.camera_tabs = CameraTabs(session)
        self.navigation_bar = NavigationBarBackNext()
        self.layout().addWidget(self.camera_tabs)
        self.layout().addWidget(self.navigation_bar)
    

class CameraTabs(QTabWidget):
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
                    tab_widgets[port] = cam_tab

            # add the widgets to the tab bar in order
            ordered_ports = list(tab_widgets.keys())
            ordered_ports.sort()
            for port in ordered_ports:
                self.insertTab(port, tab_widgets[port], f"Camera {port}")
        else:
            logger.info("No cameras available")
        
        self.toggle_tracking(self.currentIndex())
    
        
        
if __name__ == "__main__":
    from calicam import __root__
    
    App = QApplication(sys.argv)

    
    # config_path = Path(__root__, "sessions", "laptop")
    config_path = Path(__root__, "sessions", "5_cameras")
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

