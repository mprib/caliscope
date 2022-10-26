# Built following the tutorials that begin here: 
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QIcon
from PyQt6.QtWidgets import ( QVBoxLayout, QHBoxLayout, QLabel, QMainWindow, 
                            QPushButton, QTabWidget, QWidget,QGroupBox, 
                            QScrollArea, QApplication)

from pathlib import Path
from numpy import char

from qtpy import QT_VERSION

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.gui.charuco_builder import CharucoBuilder
from src.gui.camera_config_dialogue import CameraConfigDialog
from src.session import Session

class MainWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        app = QApplication.instance()
        screen = app.primaryScreen()
        DISPLAY_WIDTH = screen.size().width()
        DISPLAY_HEIGHT = screen.size().height()         
        self.setMinimumSize(DISPLAY_WIDTH*.30,DISPLAY_HEIGHT*.6)

        self.setWindowTitle("FreeMocap Camera Calibration")
        self.setWindowIcon(QIcon("src/gui/icons/fmc_logo.ico"))

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(True)
        # self.tabs.setTabsClosable(True)
        # self.tabs.tabCloseRequested.connect(self.tabs.removeTab)
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(SessionSummary(self.session), "Summary")
        # self.tabs.addTab(CharucoBuilder(self.session), "Charuco Builder")
        
        # for port, rtd in self.session.rtd.items():
            # self.tabs.addTab(CameraConfigDialog(rtd,self.session), f"Camera {port}")



class SessionSummary(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session


        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QVBoxLayout()               # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
    
        # for i in range(1,50):
        #     object = QLabel("TextLabel")
        #     self.vbox.addWidget(object)

        # self.widget.setLayout(self.vbox)

        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')
        self.show()
        # # center = QHBoxLayout()
        # # self.setLayout(center)
        # # center.addWidget(self.widget)
        # self.setMinimumHeight(1800)
        # self.scroll = QScrollArea()
        # self.widget = QWidget() 
        # self.vbox = QVBoxLayout()
        self.widget.setLayout(self.vbox)

        # self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # self.scroll.setWidgetResizable(False)
        # self.scroll.setWidget(self.widget)

        # self.setCentralWidget(self.widget)

        # # self.setLayout(self.vbox)
        

        self.build_charuco_summary()
        self.build_cam_summary()
        self.build_stereo_summary()


    def build_charuco_summary(self):
        charuco_summary = QGroupBox("Charuco Board")
        self.vbox.addWidget(charuco_summary)





        def convert_cv_qt(cv_img):
                """Convert from an opencv image to QPixmap"""
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                charuco_QImage = QImage(rgb_image.data, 
                                        w, 
                                        h, 
                                        bytes_per_line, 
                                        QImage.Format.Format_RGB888)
    
                p = charuco_QImage.scaled(self.charuco_display.width(),
                                          self.charuco_display.height(),
                                          Qt.AspectRatioMode.KeepAspectRatio, 
                                          Qt.TransformationMode.SmoothTransformation)
    
                return QPixmap.fromImage(p)
 
        pass

    def build_cam_summary(self):
        cam_summary = QGroupBox("Single Camera Calibration")
        self.vbox.addWidget(cam_summary)


        pass


    def build_stereo_summary(self):
        stereo_summary = QGroupBox("Stereocalibration")
        self.vbox.addWidget(stereo_summary) 

        pass

if __name__ == "__main__":
    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    
    app = QApplication(sys.argv)
    
    window = MainWindow(session)
    # window = SessionSummary(session)
    window.show()
    
    app.exec()