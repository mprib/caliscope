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


class MultiPlayback(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.initUI()

    def initUI(self):
        self.tabWidget = QTabWidget(self)
        self.tabWidget.setTabPosition(QTabWidget.West)  # Tabs on the left side
        self.loadTabs()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.tabWidget)
        # self.resize(1200, 800)

    def loadTabs(self):
        for port in self.controller.intrinsic_streams:
            tab = IntrinsicCalibrationWidget(controller=self.controller, port=port)
            self.tabWidget.addTab(tab, f"Cam {port}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\pyxy3d\prerecorded_workflow")
    controller = Controller(workspace_dir)
    controller.add_all_cameras_in_intrinsics_folder()
    controller.load_intrinsic_streams()

    mainWin = MultiPlayback(controller)
    mainWin.show()
    sys.exit(app.exec())
