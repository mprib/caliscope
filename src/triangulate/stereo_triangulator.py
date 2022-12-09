# this class is only a way to hold data related to the stereocamera triangulation. 
# These will load from a config file (.toml) and provide a way for the 3D triangulation
# and plotting to manage the parameters. It feels like some duplication of the camera object,
# but I want something that is designed to be simple and not actually manage the cameras, just
# organize the saved data

import logging

LOG_FILE = r"log\stereo_triangulator.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from dataclasses import dataclass

import math
from os.path import exists
import toml
import numpy as np
from pathlib import Path

import sys

sys.path.insert(0,str(Path(__file__).parent.parent.parent))
from src.triangulate.visualization.camera_mesh import CameraMesh

@dataclass
class CameraData:
    port: int
    resolution: tuple
    camera_matrix: np.ndarray
    error: float

    def __post_init__(self): 
        # initialize to origin
        self.mesh = CameraMesh(self.resolution, self.camera_matrix).mesh
        self.translation = np.array([0,0,0])
        self.rotation = np.array([[1,0,0],[0,1,0],[0,0,1]])

    def translate_mesh(self):
        scale_factor = 100
        x,y,z = [t/scale_factor for t in self.translation]
        self.mesh.translate(x,y,z)
        logging.info(f"Translating: {self.translation}")
        logging.info(f"Translation: x: {x}, y: {y}, z: {z}")

    def rotate_mesh(self):

        logging.info(f"Rotating: {self.rotation}")
        euler_angles = rotationMatrixToEulerAngles(self.rotation)
        euler_angles_deg = [x*(180/math.pi) for x in euler_angles] 
        x = euler_angles_deg[0]
        y = euler_angles_deg[1]
        z = euler_angles_deg[2]

        logging.info(f"Rotating (x,y,z euler angles): {euler_angles_deg}")
        logging.info(f"x: {x}, y: {y}, z: {z}")
        self.mesh.rotate(x,1,0,0, local=True)
        self.mesh.rotate(y,0,1,0, local=True)
        self.mesh.rotate(z,0,0,1, local=True)

 
def rotationMatrixToEulerAngles(R):
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])   
    
class StereoTriangulator:
    # created from a config.toml file, points within each camera frame can be provided to it
    # via self.locate(ArrayOfPointsA, ArrayOfPointsB)
    # perhaps I should be using pandas for some of this data processing? 
        
    def __init__(self, portA, portB, config_path):
        self.portA = portA
        self.portB = portB
        self.config_path = config_path
        
        self.load_config_data()
        
        
    def load_config_data(self):
        with open(self.config_path, "r") as f:
            logging.info(f"Loading config data located at: {self.config_path}")
            self.config = toml.load(self.config_path)

        self.camera_A = self.get_camera_at_origin(0)
        self.camera_B = self.get_camera_at_origin(1)
        
        # express location of camera B relative to Camera A
        rot, trans = self.extrinsic_params()
        self.camera_B.rotation = rot
        self.camera_B.translation = trans.squeeze() # may come in with extra dims        

        self.camera_B.translate_mesh()
        self.camera_B.rotate_mesh()

    def get_camera_at_origin(self, port):
        
        data = self.config[f"cam_{port}"]
        resolution = tuple(data["resolution"])
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        cam_data = CameraData(port, resolution, camera_matrix, data["error"]) 
        logging.info(f"Loading camera data at port {port}: {str(cam_data)}")

        return cam_data

    def extrinsic_params(self):
             
        data = self.config[f"stereo_{self.portA}_{self.portB}"]
        rotation = np.array(data["rotation"], dtype= np.float64)
        translation = np.array(data["translation"], dtype= np.float64)
        error = data["RMSE"]

        logging.info(f"Loading stereo data for ports {self.portA} and {self.portB}: {data}")
        
        return rotation, translation   
    
    
    
        
if __name__ == "__main__":

    sample_config_path  = str(Path(Path(__file__).parent, "sample_data", "config.toml"))
    triangulatr = StereoTriangulator(0,1,sample_config_path)    
    
    
    # meshA = triangulatr.get_mesh(0)
    # meshB = triangulatr.get_mesh(1)

   
