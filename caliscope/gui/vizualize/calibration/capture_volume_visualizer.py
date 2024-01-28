import caliscope.logger


import numpy as np
import pyqtgraph.opengl as gl


from caliscope.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from caliscope.cameras.camera_array import CameraArray
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume

logger = caliscope.logger.get(__name__)

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
        logger.info("refreshing capture volume scene")
        self.scene.clear()

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




# %%