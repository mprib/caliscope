from PySide6.QtWidgets import QApplication
import sys
from pathlib import Path
from caliscope.workspace_coordinator import WorkspaceCoordinator
from caliscope.gui.post_processing_widget import PostProcessingWidget

app = QApplication(sys.argv)

workspace_dir = Path("/home/mprib/caliscope_projects/markerless_calibration_data/caliscope_version")

coordinator = WorkspaceCoordinator(workspace_dir)
coordinator.load_camera_array()
window = PostProcessingWidget(coordinator)


window.show()

app.exec()
