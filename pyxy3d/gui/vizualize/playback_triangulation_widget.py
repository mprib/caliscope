import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import pyqtgraph.opengl as gl

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyxy3d.session import Session
from pyxy3d.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from pyxy3d.cameras.camera_array import CameraArray

class PlaybackTriangulationWidget(QWidget):
    def __init__(self, camera_array:CameraArray, xyz_history_path:Path):
        super(PlaybackTriangulationWidget, self).__init__()

        self.camera_array = camera_array
        self.xyz_history = pd.read_csv(xyz_history_path)

        self.visualizer = TriangulationVisualizer(self.camera_array, self.xyz_history)
        # self.visualizer.scene.show()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(self.visualizer.min_sync_index)
        self.slider.setMaximum(self.visualizer.max_sync_index)

        self.setMinimumSize(500, 500)


        self.place_widgets()
        self.connect_widgets()

        # self.visualizer.display_points(self.visualizer.min_sync_index)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)


    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)


class TriangulationVisualizer:
    """
    Can except either a single camera array or a capture volume that includes
    point_estimates. If a capture volume is supplied, point positions can
    be played back.
    """

    def __init__(
        self, camera_array: CameraArray, xyz_history:pd.DataFrame
    ):

        self.camera_array = camera_array
        self.xyz_history = xyz_history

        self.sync_indices = self.xyz_history["sync_index"]
        self.min_sync_index = np.min(self.sync_indices)
        self.max_sync_index = np.max(self.sync_indices)
        self.sync_index = self.min_sync_index

        x_coord = self.xyz_history["x_coord"]
        y_coord = self.xyz_history["y_coord"]
        z_coord = self.xyz_history["z_coord"]
        self.xyz_coord = np.vstack([x_coord,y_coord,z_coord]).T
        
        # constuct a scene
        self.scene = gl.GLViewWidget()
        self.scene.setCameraPosition(distance=4)  # the scene camera, not a real Camera

        axis = gl.GLAxisItem()
        self.scene.addItem(axis)

        # build meshes for all cameras
        self.meshes = {}
        for port, cam in self.camera_array.cameras.items():
            print(port)
            print(cam)
            mesh:CameraMesh = mesh_from_camera(cam)
            self.meshes[port] = mesh
            self.scene.addItem(mesh)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.array([0, 0, 0]),
            color=[1, 1, 1, 1],
            size=0.01,
            pxMode=False,
        )
        self.scene.addItem(self.scatter)

        self.display_points(self.sync_index)
                 
    def display_points(self, sync_index):
        """
        sync_index is provided from the dialog and linked to the slider
        it is initially set to the minimum viable sync index
        """
        self.sync_index = sync_index

        current_sync_index_flag = self.sync_indices == self.sync_index

        self.single_board_points = self.xyz_coord[current_sync_index_flag]
        logger.info(f"Displaying xyz points for sync index {sync_index}")

        self.scatter.setData(pos=self.single_board_points)

if __name__ == "__main__":

    from PyQt6.QtWidgets import QApplication

    from pyxy3d import __root__
    from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
        get_point_estimates,
    )

    from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
    import pickle

    test_recordings = [
        # Path(__root__, "dev", "sessions_copy_delete", "mediapipe_calibration", )
        Path(__root__, "dev", "sample_sessions", "recordings_to_process", "recording_4")
    ]

    test_index = 0
    recording_path = test_recordings[test_index]
    session_path = recording_path.parent
    
    logger.info(f"Loading session {session_path}")
    session = Session(session_path)

    session.load_estimated_capture_volume()

    app = QApplication(sys.argv)
    
    xyz_history_path = Path(recording_path,"point_data.csv")
    vizr_dialog = PlaybackTriangulationWidget(session.capture_volume.camera_array,xyz_history_path)
    vizr_dialog.show()

    sys.exit(app.exec())
