import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import math
import sys
import time
from pathlib import Path
from threading import Thread
from queue import Queue
import pandas as pd
import numpy as np
import pyqtgraph.opengl as gl

from random import random

from pyxy3d.cameras.camera_array import CameraData
from pyxy3d.gui.vizualize.camera_mesh import CameraMesh
from pyxy3d.cameras.camera_array_builder_deprecate import (
    CameraArray,
    CameraArrayBuilder,
)
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume


class CaptureVolumeVisualizer:
    """
    Can except either a single camera array or a capture volume that includes
    point_estimates. If a capture volume is supplied, point positions can
    be played back.
    """

    def __init__(
        self, capture_volume: CaptureVolume = None, camera_array: CameraArray = None
    ):

        if camera_array is not None and capture_volume is None:
            self.camera_array = camera_array
            self.point_estimates = None
        else:
            self.capture_volume = capture_volume
            self.camera_array = capture_volume.camera_array
            self.point_estimates = self.capture_volume.point_estimates

        # constuct a scene
        self.scene = gl.GLViewWidget()
        self.scene.setCameraPosition(distance=4)  # the scene camera, not a real Camera
        self.sync_index = None

        self.refresh_scene()

    def refresh_scene(self):
        self.scene.clear()

        axis = gl.GLAxisItem()
        self.scene.addItem(axis)

        # build meshes for all cameras
        self.meshes = {}
        for port, cam in self.camera_array.cameras.items():
            print(port)
            print(cam)
            mesh = mesh_from_camera(cam)
            self.meshes[port] = mesh
            self.scene.addItem(mesh)

        # self.scene.show()

        if self.point_estimates is not None:
            self.scatter = gl.GLScatterPlotItem(
                pos=np.array([0, 0, 0]),
                color=[1, 1, 1, 1],
                size=0.01,
                pxMode=False,
            )
            self.scene.addItem(self.scatter)

            self.sync_indices = np.unique(self.point_estimates.sync_indices)
            self.sync_indices = np.sort(self.sync_indices)

            self.min_sync_index = np.min(self.sync_indices)
            self.max_sync_index = np.max(self.sync_indices)
   
            if self.sync_index is not None:
                self.display_points(self.sync_index)
                 
    def display_points(self, sync_index):
        """
        sync_index is provided from the dialog and linked to the slider
        it is initially set to the minimum viable sync index
        """
        self.sync_index = sync_index
        current_sync_index_flag = self.point_estimates.sync_indices == sync_index
        single_board_indices = np.unique(
            self.point_estimates.obj_indices[current_sync_index_flag]
        )


        self.single_board_points = self.point_estimates.obj[single_board_indices]
        self.mean_board_position = np.mean(self.single_board_points,axis=0)
        logger.debug(f"Mean Board Position at sync index {sync_index}: {self.mean_board_position}")

        self.scatter.setData(pos=self.single_board_points)


# helper functions to assist with scene creation
def mesh_from_camera(camera_data: CameraData):
    """ "
    Mesh is placed at origin by default. Note that it appears rotations
    are in the mesh frame of reference and translations are in
    the scene frame of reference. I could be wrong, but that appears
    to be the case.

    """
    mesh = CameraMesh(camera_data.size, camera_data.matrix).mesh

    R = camera_data.rotation
    t = camera_data.translation
    camera_orientation_world = R.T

    # rotate mesh
    euler_angles = rotationMatrixToEulerAngles(camera_orientation_world)
    euler_angles_deg = [x * (180 / math.pi) for x in euler_angles]
    x = euler_angles_deg[0]
    y = euler_angles_deg[1]
    z = euler_angles_deg[2]

    # rotate mesh; z,y,x is apparently the order in which it's done
    # https://gamedev.stackexchange.com/questions/16719/what-is-the-correct-order-to-multiply-scale-rotation-and-translation-matrices-f
    mesh.rotate(z, 0, 0, 1, local=True)
    mesh.rotate(y, 0, 1, 0, local=True)
    mesh.rotate(x, 1, 0, 0, local=True)

    camera_origin_world = -np.dot(R.T, t)
    x, y, z = [p for p in camera_origin_world]
    mesh.translate(x, y, z)

    return mesh


def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# %%
if __name__ == "__main__":

    from PyQt6.QtWidgets import QApplication

    from pyxy3d import __root__
    from pyxy3d.cameras.camera_array_builder_deprecate import CameraArrayBuilder
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
    vizr = CaptureVolumeVisualizer(capture_volume=capture_volume)
    # vizr.display_points(28)
    # vizr.scene.show()
    # vizr = CaptureVolumeVisualizer(camera_array = capture_volume.camera_array)

    sys.exit(app.exec())
