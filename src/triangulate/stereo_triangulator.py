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
        self.mesh = CameraMesh(self.resolution, self.camera_matrix).mesh
        # initialize to origin
        self.translation = np.array([0,0,0])
        self.rotation = np.array([[1,0,0],[0,1,0],[0,0,1]])

 
    
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
        self.camera_B.translation = trans # may come in with extra dims        


    def get_camera_at_origin(self, port):
        
        data = self.config[f"cam_{port}"]

        resolution = tuple(data["resolution"])
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        error = data["error"]

        cam_data = CameraData(port, resolution, camera_matrix, error) 
        logging.info(f"Loading camera data at port {port}: {str(cam_data)}")

        return cam_data

    def extrinsic_params(self):
             
        data = self.config[f"stereo_{self.portA}_{self.portB}"]
        rotation = np.array(data["rotation"], dtype= np.float64).squeeze()
        translation = np.array(data["translation"], dtype= np.float64).squeeze()
        stereo_error = data["RMSE"]

        logging.info(f"Loading stereo data for ports {self.portA} and {self.portB}: {data}")
        
        return rotation, translation   
    
    
    
        
if __name__ == "__main__":

    sample_config_path  = str(Path(Path(__file__).parent, "sample_data", "config.toml"))
    triangulatr = StereoTriangulator(0,1,sample_config_path)    
    
    
    # meshA = triangulatr.get_mesh(0)
    # meshB = triangulatr.get_mesh(1)

   
