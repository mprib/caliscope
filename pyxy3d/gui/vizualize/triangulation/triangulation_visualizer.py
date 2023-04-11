import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path
import numpy as np
import pyqtgraph.opengl as gl
import pandas as pd

from pyxy3d.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.triangulate.real_time_triangulator import RealTimeTriangulator

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



# %%
if __name__ == "__main__":

    from PyQt6.QtWidgets import QApplication

    from pyxy3d import __root__
    from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
        get_point_estimates,
    )

    from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
    import pickle

    # session_directory = Path(__root__,  "tests", "2_cameras_linear")
    # session_directory = Path(__root__,  "tests", "tripod")
    # session_directory = Path(__root__,  "tests", "2_cameras_90_deg")
    # session_directory = Path(__root__,  "tests", "2_cameras_180_deg")
    # session_directory = Path(__root__,  "tests", "3_cameras_triangular")
    # session_directory = Path(__root__,  "tests", "3_cameras_middle")
    # session_directory = Path(__root__, "tests", "4_cameras_beginning")
    # session_directory = Path(__root__, "tests", "4_cameras_endofday")
    session_directory = Path(__root__,  "tests", "4_cameras_nonoverlap")
    # session_directory = Path(__root__,  "tests", "3_cameras_linear")
    # session_directory = Path(__root__,  "tests", "3_cameras_midlinear")
    # session_directory = Path(__root__,  "tests", "just_checking")

    saved_CV_path = Path(session_directory, "capture_volume_stage_1_optimized.pkl")
    # saved_CV_path = Path(session_directory, "capture_volume_stage_1_new_origin.pkl")
    with open(saved_CV_path, "rb") as f:
        capture_volume: CaptureVolume = pickle.load(f)

    app = QApplication(sys.argv)
    with open(saved_CV_path, "rb") as f:
        capture_volume: CaptureVolume = pickle.load(f)

    app = QApplication(sys.argv)
    vizr = TriangulationVisualizer(capture_volume=capture_volume)
    # vizr.display_points(28)
    # vizr.scene.show()
    # vizr = CaptureVolumeVisualizer(camera_array = capture_volume.camera_array)

    sys.exit(app.exec())
