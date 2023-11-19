import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget
from pathlib import Path
from pyxy3d.gui.prerecorded_intrinsic_calibration.playback_widget import IntrinsicCalibrationWidget
from pyxy3d.controller import Controller

class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.initUI()

    def initUI(self):
        self.tabWidget = QTabWidget(self)
        self.tabWidget.setTabPosition(QTabWidget.West)  # Tabs on the left side

        self.loadTabs()

        self.setCentralWidget(self.tabWidget)
        self.setWindowTitle("Intrinsic Calibration Tabs")
        self.resize(1200, 800)

    def loadTabs(self):
        # Assuming controller.load_intrinsic_streams() loads streams and 
        # controller.streams is an iterable of stream IDs

        for port in self.controller.intrinsic_streams:
            tab = IntrinsicCalibrationWidget(controller=self.controller, port=port)
            self.tabWidget.addTab(tab, f"Stream {port}")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\prerecorded_workflow")
    controller = Controller(workspace_dir)
    controller.add_all_cameras_in_intrinsics_folder()
    controller.load_intrinsic_streams()

    mainWin = MainWindow(controller)
    mainWin.show()
    sys.exit(app.exec())
