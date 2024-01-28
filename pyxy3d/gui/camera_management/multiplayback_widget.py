import sys
from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QTabWidget,
    QWidget,
)
from pathlib import Path
from caliscope.gui.camera_management.playback_widget import (
    IntrinsicCalibrationWidget,
)
from caliscope.controller import Controller
import caliscope.logger

logger = caliscope.logger.get(__name__)


class MultiIntrinsicPlaybackWidget(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.initUI()

    def initUI(self):
        self.tabWidget = QTabWidget(self)
        self.tabWidget.setTabPosition(QTabWidget.South)  # Tabs on the left side
        self.loadTabs()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.tabWidget)
        # self.resize(1200, 800)

    def loadTabs(self):
        logger.info("Beginning to load individual tabs")
        for camera in self.controller.camera_array.cameras.values():
            logger.info(f"About to create calibration widget for camera {camera.port}")
            tab = IntrinsicCalibrationWidget(
                controller=self.controller, port=camera.port
            )
            logger.info(f"Calibration widget for camera {camera.port} successfully created")
            self.tabWidget.addTab(tab, f"Cam {camera.port}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\caliscope\prerecorded_workflow")
    controller = Controller(workspace_dir)
    controller.load_camera_array()
    controller.load_intrinsic_stream_manager()

    mainWin = MultiIntrinsicPlaybackWidget(controller)
    mainWin.show()
    sys.exit(app.exec())
