# Built following the tutorials that begin here: 
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/

from re import L
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
        self.summary = SessionSummary(self.session)

        self.tabs.addTab(self.summary, "Summary")
        self.summary.launch_charuco_builder_btn.clicked.connect(self.launch_cha_build)
        # self.tabs.addTab(CharucoBuilder(self.session), "Charuco Builder")
        # self.su 
        # for port, rtd in self.session.rtd.items():
            # self.tabs.addTab(CameraConfigDialog(rtd,self.session), f"Camera {port}")

    def test_function(self):
        print("working")
        self.summary.update_charuco_summary()
    
    def launch_cha_build(self):
        # check to see if it exists
        for t in range(0,self.tabs.count()):
            if self.tabs.tabText(t) == "Charuco Builder":
                return

        self.charuco_builder = CharucoBuilder(self.session)
        self.charuco_builder.export_btn.clicked.connect(self.test_function)
        self.tabs.addTab(self.charuco_builder, "Charuco Builder")
        # self.tabs["Charuco Builder"].setClosable(True)

class SessionSummary(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session


        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QVBoxLayout()               # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
    
        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')
        self.show()

        self.widget.setLayout(self.vbox)

        # realizing that it is important to place widgets in init so that the
        # data can be refreshed from an update() call and the layout will
        # remain unchanged        
        self.charuco_summary = QGroupBox("Charuco Board")
        self.vbox.addWidget(self.charuco_summary)

        self.cam_summary = QGroupBox("Single Camera Calibration")
        self.vbox.addWidget(self.cam_summary)

        self.build_charuco_summary()
        self.build_cam_summary()
        self.build_stereo_summary()


    def build_charuco_summary(self):
        # self.charuco_summary.setLayout(None)
        self.charuco_hbox = QHBoxLayout()
        self.charuco_summary.setLayout(self.charuco_hbox)
        self.charuco_display = QLabel()
        self.charuco_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)       

        self.charuco_hbox.addWidget(self.charuco_display)

        right_vbox = QVBoxLayout()
        self.charuco_summary = QLabel()

        self.launch_charuco_builder_btn = QPushButton("Launch Charuco Builder")        
        self.launch_charuco_builder_btn.setMaximumSize(150,50)
        
        right_vbox.addWidget(self.charuco_summary)
        right_vbox.addWidget(self.launch_charuco_builder_btn)

        right_vbox.setAlignment(self.charuco_summary, Qt.AlignmentFlag.AlignHCenter) 
        right_vbox.setAlignment(self.charuco_display, Qt.AlignmentFlag.AlignHCenter) 

        self.charuco_hbox.addLayout(right_vbox)

        self.update_charuco_summary()

    def update_charuco_summary(self):
        charuco_width = self.width()/4
        charuco_height = self.height()/4
        charuco_img = self.session.charuco.board_pixmap(charuco_width, charuco_height)
        self.charuco_display.setPixmap(charuco_img)
        self.charuco_summary.setText(self.session.charuco.summary())
        


    def build_cam_summary(self):
        self.cam_hbox = QHBoxLayout()
        self.cam_summary.setLayout(self.cam_hbox)

        left_vbox = QVBoxLayout()

        self.connect_cameras_btn = QPushButton("Connect Cameras")
        # self.find_cameras_btn = QPushButton("Find Additional Cameras")
        self.open_cameras_btn = QPushButton("Open Camera Calibration") 
        left_vbox.addWidget(self.connect_cameras_btn)
        left_vbox.addWidget(self.open_cameras_btn)

        

    def build_stereo_summary(self):
        stereo_summary = QGroupBox("Stereocalibration")
        self.vbox.addWidget(stereo_summary) 


if __name__ == "__main__":
    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    
    app = QApplication(sys.argv)
    
    window = MainWindow(session)
    # window = SessionSummary(session)
    window.show()
    
    app.exec()