from PyQt6.QtWidgets import QMainWindow, QSizePolicy, QApplication, QGridLayout, QWidget, QHBoxLayout
import sys
from pathlib import Path
import time

sys.path.insert(0,str(Path(__file__).parent.parent.parent))
from src.session import Session
from src.gui.config_tree import ConfigTree
from src.gui.camera_config_dialogue import CameraConfigDialog
from src.gui.charuco_builder import CharucoBuilder

class CalibrationInterface(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.setMinimumHeight(int(DISPLAY_HEIGHT*.5))
        self.setMinimumWidth(int(DISPLAY_HEIGHT*.5))
        # self.setMinimumSize(int(DISPLAY_WIDTH*.5),int(DISPLAY_HEIGHT*.5))
        main = QWidget()
        self.session = session 
        self.grid = QGridLayout() 
        main.setLayout(self.grid)
        self.setCentralWidget(main)
        self.setWindowTitle("Camera Calibration")

        self.config_tree = ConfigTree(self.session)
        # config_tree.treeView.setFixedHeight(self.height())
        # main.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)
        self.config_tree.treeView.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.grid.addWidget(self.config_tree, 0, 0)

        self.config_tree.treeView.doubleClicked.connect(self.getValue)
        
        self.charuco_builder = CharucoBuilder(self.session)

    def getValue(self, val):
        print(val.parent().data())
        print(val.data())

        if val.parent().data() == "Charuco":
            self.load_charuco()

    def load_charuco(self):
        
        self.grid.addWidget(self.charuco_builder, 0, 1)
        # this is what allows the tree to update based on changes to the charuco
        def on_export_btn():
            self.config_tree = None
            self.config_tree = ConfigTree(self.session)
            self.grid.addWidget(self.config_tree, 0, 0)

        self.charuco_builder.export_btn.clicked.connect(on_export_btn)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')

    screen = app.primaryScreen()
    DISPLAY_WIDTH = screen.size().width()
    DISPLAY_HEIGHT = screen.size().height()

    calibration_ui = CalibrationInterface(session)

    calibration_ui.show()

    sys.exit(app.exec())