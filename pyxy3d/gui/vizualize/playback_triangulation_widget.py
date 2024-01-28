from pathlib import Path
import numpy as np
from time import time
import pyqtgraph.opengl as gl

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSlider,
    QVBoxLayout,
    QWidget,
)
from caliscope.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from caliscope.cameras.camera_array import CameraArray
import caliscope.logger

from caliscope.motion_trial import MotionTrial

logger = caliscope.logger.get(__name__)
# as part of development process I'm just going to import the skeleton in here

class PlaybackTriangulationWidget(QWidget):
    def __init__(self, camera_array: CameraArray, xyz_history_path: Path = None):
        super(PlaybackTriangulationWidget, self).__init__()

        self.camera_array = camera_array
        self.visualizer = TriangulationVisualizer(self.camera_array)
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self.setMinimumSize(500, 500)

        self.place_widgets()
        self.connect_widgets()
        if xyz_history_path is not None:
            self.update_motion_trial(xyz_history_path)
        else:
            self.motion_trial = None

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)

    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)
        self.slider.valueChanged.connect(self.visualizer.update_segment_lines)

    def update_motion_trial(self, xyz_history_path):
        # self.xyz_history = pd.read_csv(xyz_history)
        tic = time()
        logger.info(f"Beginning to load in motion trial: {time()}")
        self.motion_trial = MotionTrial(xyz_history_path)
        logger.info(f"Motion trial loading complete: {time()} ")
        toc = time()
        logger.info(f"Elapsed time to load: {toc-tic}")

        self.visualizer.update_motion_trial(self.motion_trial)

        if self.motion_trial.is_empty:
            self.slider.setMinimum(0)
            self.slider.setMaximum(100)
        else:
            self.slider.setMinimum(self.motion_trial.start_index)
            self.slider.setMaximum(self.motion_trial.end_index)

    def update_camera_array(self, camera_array: CameraArray):
        self.visualizer.update_camera_array(camera_array)


class TriangulationVisualizer:

    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.build_scene()

    def build_scene(self):
        # constuct a scene if not yet there
        if hasattr(self, "scene"):
            logger.info("Clearing scene in capture volume visualizer")
            self.scene.clear()
        else:
            logger.info("Creating initial scene in capture volume visualizer")
            self.scene = gl.GLViewWidget()

            # the scene camera, not a real Camera
            self.scene.setCameraPosition( distance=4)  
        axis = gl.GLAxisItem()
        self.scene.addItem(axis)

        if self.camera_array.all_extrinsics_calibrated():
            self.meshes = {}
            for port, cam in self.camera_array.cameras.items():
                mesh: CameraMesh = mesh_from_camera(cam)
                self.meshes[port] = mesh
                self.scene.addItem(mesh)

        self.scatter = gl.GLScatterPlotItem(
            pos=np.array([0, 0, 0]),
            color=[1, 1, 1, 1],
            size=0.01,
            pxMode=False,
        )
        
            # will hold a list
        self.segments = {}
        
        self.scene.addItem(self.scatter)
        self.scatter.setData(pos=None)

    def update_camera_array(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.build_scene()

    def update_motion_trial(self, motion_trial:MotionTrial):
        logger.info("Updating xyz history in playback widget")
        self.motion_trial:MotionTrial = motion_trial
        
        if hasattr(self.motion_trial.tracker, "wireframe"):
            for segment_line in self.motion_trial.tracker.wireframe.line_plots.values():
                self.scene.addItem(segment_line)

        self.sync_index = self.motion_trial.start_index
        self.display_points(self.sync_index)


    def display_points(self, sync_index:int):
        """
        sync_index is provided from the dialog and linked to the slider
        it is initially set to the minimum viable sync index
        """

        if self.motion_trial.is_empty:
            self.scatter.setData(pos=None)
        else:
            self.sync_index = sync_index
            logger.debug(f"Displaying xyz points for sync index {sync_index}")
            xyz_coords = self.motion_trial.get_xyz(self.sync_index).point_xyz
            self.scatter.setData(pos=xyz_coords)


    def update_segment_lines(self,sync_index:int):
        self.motion_trial.update_wireframe(sync_index)