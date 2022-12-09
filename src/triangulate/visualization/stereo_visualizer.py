
import logging
import pprint

LOG_FILE = r"log\stereo_visualizer.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
import math
import numpy as np
from pathlib import Path
import pyqtgraph as pg
import pyqtgraph.opengl as gl

sys.path.insert(0,str(Path(__file__).parent.parent.parent.parent))

from src.triangulate.visualization.camera_mesh import CameraMesh
from src.triangulate.stereo_triangulator import CameraData

class StereoVisualizer:
    
    def __init__(self, triangulator):
        self.triangulator = triangulator
        
        self.mesh_A = mesh_from_camera(triangulator.camera_A)
        self.mesh_B = mesh_from_camera(triangulator.camera_B)

        # create the overhead for display 
        self.app = pg.mkQApp("Stereo Visualizer")
        self.scene = gl.GLViewWidget()
        self.scene.setWindowTitle('Camera Calibration')
        self.scene.setCameraPosition(distance=4)
        
        grid = gl.GLGridItem()
        grid.scale(1,1,1)

        self.scene.addItem(grid)
        self.scene.addItem(self.mesh_A)
        self.scene.addItem(self.mesh_B)

        self.scene.show()
        pg.exec()
        
    def show(self):
        pass

# helper functions to assist with scene creation
def mesh_from_camera(cd: CameraData):
    # cd = camera_data
    mesh = CameraMesh(cd.resolution, cd.camera_matrix).mesh
    
    # translate mesh which defaults to origin
    translation_scale_factor = 100
    x,y,z = [t/translation_scale_factor for t in cd.translation]
    mesh.translate(x,y,z)
    logging.info(f"Translation: x: {x}, y: {y}, z: {z}")

    # rotate mesh
    logging.info(f"Rotating: {cd.rotation}")
    euler_angles = rotationMatrixToEulerAngles(cd.rotation)
    euler_angles_deg = [x*(180/math.pi) for x in euler_angles] 
    x = euler_angles_deg[0]
    y = euler_angles_deg[1]
    z = euler_angles_deg[2]

    logging.info(f"x: {x}, y: {y}, z: {z}")
    mesh.rotate(x,1,0,0, local=True)
    mesh.rotate(y,0,1,0, local=True)
    mesh.rotate(z,0,0,1, local=True)

    return mesh
          
def rotationMatrixToEulerAngles(R):
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])   

if __name__ == '__main__':

    from src.triangulate.stereo_triangulator import StereoTriangulator
    
    sample_config_path  = str(Path(Path(__file__).parent.parent, "sample_data", "config.toml"))
    trigr = StereoTriangulator(0,1,sample_config_path)    

    vizr = StereoVisualizer(trigr)