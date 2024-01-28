# %%

import caliscope.logger

import numpy as np
from dataclasses import dataclass
import cv2
from enum import Enum, auto
from numba.typed import Dict
from collections import OrderedDict
logger = caliscope.logger.get(__name__)
CAMERA_PARAM_COUNT = 6

@dataclass
class CameraData:
    """
    A place to hold the calibration data associated with a camera that has been populated from a config file.
    For use with final setting of the array and triangulation, but no actual camera management.
    Not loving the way this is implemented as an adjunct to the Camera object, but here we are
    """

    port: int
    size: tuple
    rotation_count: int=0
    error: float=None  # the RMSE of reprojection associated with the intrinsic calibration
    matrix: np.ndarray=None
    distortions: np.ndarray=None  #
    exposure: int=None
    grid_count: int=None
    ignore: bool=False
    verified_resolutions: np.ndarray=None
    translation: np.ndarray =None # camera relative to world
    rotation: np.ndarray = None# camera relative to world

    @property
    def transformation(self):
        """"
        Rotation and transformation combined to allow 
        """
        
        t = np.hstack([self.rotation, np.expand_dims(self.translation, 1)])
        t = np.vstack([t, np.array([0,0,0,1], np.float32)])
        return t 
   
    @transformation.setter 
    def transformation(self, t: np.ndarray):
        self.rotation = t[0:3,0:3]
        self.translation = t[0:3,3]
        logger.info(f"Rotation and Translation being updated to {self.rotation} and {self.translation}")
    
    @property  
    def projection_matrix(self):
        return self.matrix @ self.transformation[0:3,:]
         
    def extrinsics_to_vector(self):
        """
        Converts camera parameters to a numpy vector for use with bundle adjustment.
        """
        # rotation of the camera relative to the world
        rotation_rodrigues = cv2.Rodrigues(self.rotation)[0]  # elements 0,1,2
        port_param = np.hstack([rotation_rodrigues[:, 0], self.translation])

        return port_param

    def extrinsics_from_vector(self, row):
        """
        Takes a vector in the same format that is output of `extrinsics_to_vector` and updates the camera
        """

        # convert back to world frame of reference
        self.rotation = cv2.Rodrigues(row[0:3])[0]
        self.translation = np.array([row[3:6]], dtype=np.float64)[0]

    def get_display_data(self)-> OrderedDict:
        
        # Extracting camera matrix parameters
        # self.matrix = None
        if self.matrix is not None:
            fx, fy = self.matrix[0, 0], self.matrix[1, 1]
            cx, cy = self.matrix[0, 2], self.matrix[1, 2]
        else:
            fx, fy = None, None
            cx, cy = None, None
            

        # Extracting distortion coefficients
        if self.distortions is not None:
            k1, k2, p1, p2, k3 = self.distortions.ravel()[:5]
        else:
            k1, k2, p1, p2, k3 = None, None, None, None, None

        def round_or_none(value,places):
            if value is None:
                return None
            else:
                return round(value,places)
            
        # Creating the dictionary with OrderedDict
        camera_display_dict = OrderedDict([
            ("size", self.size),
            ("RMSE", self.error),
            ("Grid_Count", self.grid_count),
            ("rotation_count", self.rotation_count),
            ("intrinsic_parameters", OrderedDict([
                ("focal_length_x", round_or_none(fx,2)),
                ("focal_length_y", round_or_none(fy,2)),
                ("optical_center_x", round_or_none(cx,2)),
                ("optical_center_y", round_or_none(cy,2))
            ])),
            ("distortion_coefficients", OrderedDict([
                ("radial_k1", round_or_none(k1,2)),
                ("radial_k2", round_or_none(k2,2)),
                ("radial_k3", round_or_none(k3,2)),
                ("tangential_p1", round_or_none(p1,2)),
                ("tangential_p2", round_or_none(p2,2))
            ]))
        ])

        return camera_display_dict


    def erase_calibration_data(self):
        self.error = None
        self.matrix = None
        self.distortions = None
        self.grid_count = None
        self.translation = None
        self.rotation = None
        
@dataclass
class CameraArray:
    """The plan is that this will expand to become an interface for setting the origin.
    At the moment all it is doing is holding a dictionary of CameraData objects"""

    cameras: dict

    @property
    def port_index(self):
        """
        Provides a dictionary mapping the camera port to an index. Generally,this
        will match the camera ports 1:1, but will be different when a camera 
        is being ignored. Used to manage reference to camera parameters in xy_reprojection_error
        used within the least_squares optimization of the capture volume. 
        """
        not_ignored_ports = [port for port, cam in self.cameras.items() if not cam.ignore]
        not_ignored_ports.sort()
        not_ignored_indices = [i for i in range(len(not_ignored_ports))]
        port_indices = {port:i for port, i in zip(not_ignored_ports,not_ignored_indices)}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        
        return port_indices
   
    @property 
    def index_port(self):
        return {value: key for key, value in self.port_index.items()}
    
    def get_extrinsic_params(self):
        """for each camera build the CAMERA_PARAM_COUNT element parameter index
        camera_params with shape (n_cameras, CAMERA_PARAM_COUNT)
        contains initial estimates of parameters for all cameras.
        First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
        next 3 components form a translation vector
        """

        camera_params = None
        # ensure that parameters are built up in order of the corresponding index
        for index in sorted(self.index_port.keys()):
            port = self.index_port[index]
            cam = self.cameras[port]
            port_param = cam.extrinsics_to_vector()

            if camera_params is None:
                camera_params = port_param
            else:
                camera_params = np.vstack([camera_params, port_param])

        return camera_params

    def update_extrinsic_params(self, least_sq_result_x:np.array):

        n_cameras = len(self.port_index)
        n_cam_param = 6  # 6 DoF
        flat_camera_params = least_sq_result_x[0 : n_cameras * n_cam_param]
        new_camera_params = flat_camera_params.reshape(n_cameras, n_cam_param)

        # update camera array with new positional data
        for index in range(len(new_camera_params)):
            port = self.index_port[index]  # correct in case ignoring a camera
            cam_vec = new_camera_params[index, :]
            self.cameras[port].extrinsics_from_vector(cam_vec)
        
    def all_extrinsics_calibrated(self)->bool:
        # assume extrinsics calibrated and provide otherwise
        full_extrinsics = True 
        for port, cam in self.cameras.items():
            if cam.rotation is None or cam.translation is None:
                full_extrinsics = False
        return full_extrinsics
   
    def all_intrinsics_calibrated(self)->bool:
        # assume true and prove false
        full_intrinsics = True
        for port, cam in self.cameras.items():
            if cam.matrix is None or cam.distortions is None:
                full_intrinsics = False
        return full_intrinsics
    
    @property
    def projection_matrices(self) -> Dict: 
        logger.info("Creating camera array projection matrices")
        proj_mat = Dict()
        for port, cam in self.cameras.items():
            proj_mat[port] = cam.projection_matrix
        
        return proj_mat
            
