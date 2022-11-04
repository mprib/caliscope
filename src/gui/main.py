# Built following the tutorials that begin here: 
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QIcon
from PyQt6.QtWidgets import ( QVBoxLayout, QHBoxLayout, QLabel, QMainWindow, 
                            QPushButton, QTabWidget, QWidget,QGroupBox, 
                            QScrollArea, QApplication, QTableWidget, QSizePolicy)

from pathlib import Path
from numpy import char
from threading import Thread

from qtpy import QT_VERSION

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gui.charuco_builder import CharucoBuilder
from src.gui.camera_config_dialogue import CameraConfigDialog
from src.session import Session
from camera_table import CameraTable


class MainWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        app = QApplication.instance()
        screen = app.primaryScreen()
        DISPLAY_WIDTH = screen.size().width()
        DISPLAY_HEIGHT = screen.size().height()         
        self.setMinimumSize(DISPLAY_WIDTH*.30,DISPLAY_HEIGHT*.7)

        self.setWindowTitle("FreeMocap Camera Calibration")
        self.setWindowIcon(QIcon("src/gui/icons/fmc_logo.png"))
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(True)

        self.setCentralWidget(self.tabs)
        self.summary = SessionSummary(self.session)
        self.tabs.addTab(self.summary, "Summary")
        
        self.summary.launch_charuco_builder_btn.clicked.connect(self.launch_cha_build)
        self.summary.open_cameras_btn.clicked.connect(self.open_cams)
        self.summary.close_cameras_btn.clicked.connect(self.close_cams)

    def open_cams(self):
        # see if this helps with updating changes
        self.session.load_config()
        self.session.load_charuco()

        # don't bother if already done
        for t in range(0,self.tabs.count()):
            if self.tabs.tabText(t).startswith("Cam"):
                return
            
        if len(self.session.rtd) > 0:
            for port, rtd in self.session.rtd.items():
                
                cam_tab = CameraConfigDialog(rtd, self.session)
                
                self.tabs.addTab(cam_tab, f"Camera {port}")
                cam_tab.save_cal_btn.clicked.connect(self.summary.camera_table.update_data)
        else:
            print("No cameras available")

    def close_cams(self):
        print("Attempting to close cameras")
        tab_count = self.tabs.count()
        for t in range(tab_count,0,-1):
            if self.tabs.tabText(t).startswith("Cam"):
                self.tabs.removeTab(t)



    def update_summary_image(self):
        self.summary.update_charuco_summary()
    
    def launch_cha_build(self):
        # check to see if it exists
        for t in range(0,self.tabs.count()):
            if self.tabs.tabText(t) == "Charuco Builder":
                return

        self.charuco_builder = CharucoBuilder(self.session)
        self.charuco_builder.export_btn.clicked.connect(self.update_summary_image)
        self.tabs.addTab(self.charuco_builder, "Charuco Builder")
        # self.tabs["Charuco Builder"].setClosable(True)

class SessionSummary(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session

        self.cams_connected = False

        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QVBoxLayout()               # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.setCentralWidget(self.scroll)

        # self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')
        self.show()

        self.widget.setLayout(self.vbox)

        # realizing that it is important to place widgets in init so that the
        # data can be refreshed from an update() call and the layout will
        # remain unchanged       
        self.top_hbox = QHBoxLayout()
        self.vbox.addLayout(self.top_hbox)
        self.vbox.setAlignment(self.top_hbox, Qt.AlignmentFlag.AlignTop) 
        self.charuco_summary = QGroupBox("Charuco Board")
        self.charuco_summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.top_hbox.addWidget(self.charuco_summary)

        self.cam_summary = QGroupBox("Single Camera Calibration")
        self.top_hbox.addWidget(self.cam_summary)
        self.top_hbox.setAlignment(self.cam_summary, Qt.AlignmentFlag.AlignTop)       
        self.build_charuco_summary()
        self.build_cam_summary()
        self.build_stereo_summary()


    def build_charuco_summary(self):
        # self.charuco_summary.setLayout(None)
        self.charuco_hbox = QHBoxLayout()
        self.charuco_summary.setLayout(self.charuco_hbox)
        # self.charuco_summary.setSizePolicy(QSizePolicy.verticalPolicy)
        self.charuco_display = QLabel()
        self.charuco_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.charuco_display.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)       
        self.charuco_hbox.addWidget(self.charuco_display)
        self.charuco_hbox.setAlignment(self.charuco_display, Qt.AlignmentFlag.AlignBottom)

        right_vbox = QVBoxLayout()
        self.charuco_summary = QLabel()

        self.launch_charuco_builder_btn = QPushButton("Launch Charuco Builder")        
        self.launch_charuco_builder_btn.setMaximumSize(150,30)
        
        right_vbox.addWidget(self.charuco_summary)
        right_vbox.addWidget(self.launch_charuco_builder_btn)
        right_vbox.setAlignment(self.launch_charuco_builder_btn,Qt.AlignmentFlag.AlignBottom)
        self.charuco_hbox.addLayout(right_vbox)
        self.charuco_hbox.setAlignment(right_vbox, Qt.AlignmentFlag.AlignBottom) 
        # right_vbox.setAlignment(self.charuco_display, Qt.AlignmentFlag.AlignHCenter) 


        self.update_charuco_summary()

    def update_charuco_summary(self):
        charuco_width = 200
        charuco_height = 200
        charuco_img = self.session.charuco.board_pixmap(charuco_width, charuco_height)
        self.charuco_display.setPixmap(charuco_img)
        self.charuco_summary.setText(self.session.charuco.summary())
        

    def find_connect_cams(self):
        
        def find_cam_worker():

            self.session.load_cameras()
            self.session.find_additional_cameras()
            self.session.load_rtds()
            self.session.adjust_resolutions()
            self.camera_table.update_data()

        if not self.cams_connected:
            print("Connecting to cameras...This may take a moment.")
            self.find_cams = Thread(target=find_cam_worker, args=(), daemon=True)
            self.find_cams.start()
        else:
            print("Cameras already connected or in process.")

        self.cams_connected = True

    def build_cam_summary(self):
        self.cam_hbox = QHBoxLayout()
        self.cam_summary.setLayout(self.cam_hbox)

        left_vbox = QVBoxLayout()

        self.camera_table = CameraTable(self.session)
        # self.camera_table.setFixedSize(self.width(),self.height() )
        self.camera_table.setFixedSize(250, 150)
        # self.camera_table.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        left_vbox.addWidget(self.camera_table)
        self.find_connect_cams_btn = QPushButton("Find and Connect to Cameras")


        self.find_connect_cams_btn.clicked.connect(self.find_connect_cams)
        left_vbox.addWidget(self.find_connect_cams_btn)

        self.open_cameras_btn = QPushButton("Open Cameras") 
        self.close_cameras_btn = QPushButton("Close Cameras")
        # self.open_cameras_btn.clicked.connect(open_cams)

        left_vbox.addWidget(self.open_cameras_btn)
        left_vbox.addWidget(self.close_cameras_btn)
        self.cam_hbox.addLayout(left_vbox)



    
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