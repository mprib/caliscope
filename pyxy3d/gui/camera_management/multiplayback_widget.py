import sys
from PySide6.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QTabWidget,
    QWidget,
)
from pathlib import Path
from pyxy3d.gui.prerecorded_intrinsic_calibration.playback_widget import (
    IntrinsicCalibrationWidget,
)
from pyxy3d.controller import Controller


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
        for camera in self.controller.camera_array.cameras.values():
            tab = IntrinsicCalibrationWidget(controller=self.controller, port=camera.port)
            self.tabWidget.addTab(tab, f"Cam {camera.port}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\prerecorded_workflow")
    controller = Controller(workspace_dir)
    controller.load_camera_array()
    controller.load_intrinsic_stream_manager()

    mainWin = MultiIntrinsicPlaybackWidget(controller)
    mainWin.show()
    sys.exit(app.exec())
