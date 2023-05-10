
#%%

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from pathlib import Path
from datetime import datetime
from os.path import exists
import numpy as np
import toml
from dataclasses import asdict

from pyxy3d.calibration.charuco import Charuco
from pyxy3d.cameras.camera import Camera
from pyxy3d.cameras.live_stream import LiveStream
from pyxy3d.interface import TrackerEnum
from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates, load_point_estimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from concurrent.futures import ThreadPoolExecutor

class Configurator:
    """
    A factory to provide pre-configured objects and to save the configuration
    of currently existing objects. 
    """    
    def __init__(self, session_path:Path) -> None:
        self.session_path = session_path
        self.toml_path = Path(self.session_path,"config.toml")
        
        if exists(self.toml_path):
            logger.info("Found previous config")
            with open(self.toml_path, "r") as f:
                self.dict = toml.load(self.toml_path)
        else:
            logger.info(
                "No existing config.toml found; creating starter file with charuco"
            )

            self.dict = toml.loads("")
            self.dict["CreationDate"] = datetime.now()
            with open(self.toml_path, "a") as f:
                toml.dump(self.dict, f)


    def update_toml(self):
        # alphabetize by key to maintain standardized layout
        sorted_dict = {key: value for key, value in sorted(self.dict.items())}
        self.dict = sorted_dict

        with open(self.toml_path, "w") as f:
            toml.dump(self.dict, f)

    def get_capture_volume(self)->CaptureVolume:
        camera_array = self.get_camera_array()
        point_estimates = self.get_point_estimates()
        
        capture_volume = CaptureVolume(camera_array,point_estimates)
        return capture_volume
    
    def save_capture_volume(self, capture_volume:CaptureVolume):
        # self.point_estimates = self.capture_volume.point_estimates
        # self.camera_array = self.capture_volume.camera_array
        self.save_camera_array(capture_volume.camera_array)
        self.save_point_estimates(capture_volume.point_estimates)

        self.dict["capture_volume"] = {}
        # self["capture_volume"]["RMSE_summary"] = self.capture_volume.rmse
        self.dict["capture_volume"]["stage"] = capture_volume.stage
        self.dict["capture_volume"][
            "origin_sync_index"
        ] = capture_volume.origin_sync_index
        self.update_toml()
        
     
    def get_camera_array(self)->CameraArray:
        """
        Load camera array directly from config file. The results of capture volume
        optimization and origin transformation will be reflected in this array
        which can then be the basis for future 3d point estimation
        """
        all_camera_data = {}
        for key, params in self.dict.items():
            if key.startswith("cam"):
                if params["ignore"] == False:
                    port = params["port"]

                    if "error" in params.keys(): #intrinsics have been calculated
                        error = params["error"]
                        matrix = np.array(params["matrix"])
                        distortions = np.array(params["distortions"])
                        grid_count = params["grid_count"]
                    else: 
                        error = None
                        matrix = None
                        distortions = None
                        grid_count = None

                    if "translation" in params.keys(): #Extrinsics have been calculated
                        translation = np.array(params["translation"])
                        rotation = np.array(params["rotation"])
                    else:
                        translation = None
                        rotation = None

                    logger.info(f"Adding camera {port} to calibrated camera array...")
                    cam_data = CameraData(
                        port=port,
                        size=params["size"],
                        rotation_count=params["rotation_count"],
                        error= error,
                        matrix=matrix,
                        distortions=distortions,
                        exposure=params["exposure"],
                        grid_count=grid_count,
                        ignore=params["ignore"],
                        verified_resolutions=params["verified_resolutions"],
                        translation=translation,
                        rotation=rotation
                    )

                    all_camera_data[port] = cam_data

        camera_array = CameraArray(all_camera_data)
        return camera_array
    
    
    def get_point_estimates(self)->PointEstimates:
        point_estimates_dict = self.dict["point_estimates"]

        for key, value in point_estimates_dict.items():
            point_estimates_dict[key] = np.array(value)

        point_estimates = PointEstimates(**point_estimates_dict)
        return point_estimates
    
    def get_charuco(self):
        if "charuco" in self.dict:
            logger.info("Loading charuco from config")
            params = self.dict["charuco"]

            charuco = Charuco(
                columns=params["columns"],
                rows=params["rows"],
                board_height=params["board_height"],
                board_width=params["board_width"],
                dictionary=params["dictionary"],
                units=params["units"],
                aruco_scale=params["aruco_scale"],
                square_size_overide_cm=params["square_size_overide_cm"],
                inverted=params["inverted"],
            )
        else:
            logger.info("Loading default charuco")
            charuco = Charuco(4, 5, 11, 8.5, square_size_overide_cm=5.4)
            self.save_charuco(charuco)

        return charuco

    
    def save_charuco(self, charuco:Charuco):
        self.dict["charuco"] = charuco.__dict__
        logger.info(f"Saving charuco with params {charuco.__dict__} to config")
        self.update_toml()

    def save_camera(self, camera: Camera | CameraData):
        def none_or_list(value):
            # required to make sensible numeric format
            # otherwise toml formats as text
            if value is None:
                return None
            else:
                return value.tolist()

        params = {
            "port": camera.port,
            "size": camera.size,
            "rotation_count": camera.rotation_count,
            "error": camera.error,
            "matrix": none_or_list(camera.matrix),
            "distortions": none_or_list(camera.distortions),
            "translation": none_or_list(camera.translation),
            "rotation": none_or_list(camera.rotation),
            "exposure": camera.exposure,
            "grid_count": camera.grid_count,
            "ignore": camera.ignore,
            "verified_resolutions": camera.verified_resolutions,
        }

        self.dict["cam_" + str(camera.port)] = params
        self.update_toml()

    def save_camera_array(self, camera_array:CameraArray):
        logger.info("Saving camera array....")
        for port, camera_data in camera_array.cameras.items():
            camera_data = camera_array.cameras[port]
            self.save_camera(camera_data)

    def get_cameras(self)-> dict[Camera]:
        cameras = {}
        def add_preconfigured_cam(params):
            # try:
            port = params["port"]
            logger.info(f"Attempting to add pre-configured camera at port {port}")

            if params["ignore"]:
                logger.info(f"Ignoring camera at port {port}")
                pass  # don't load it in
            else:
                if "verified_resolutions" in params.keys():
                    verified_resolutions = params["verified_resolutions"]
                    cameras[port] = Camera(port, verified_resolutions)
                else:
                    cameras[port] = Camera(port)

                camera = cameras[port]  # just for ease of reference
                camera.rotation_count = params["rotation_count"]
                camera.exposure = params["exposure"]

                # if calibration done, then populate those as well
                if "error" in params.keys():
                    logger.info(f"Camera RMSE error for port {port}: {params['error']}")
                    camera.error = params["error"]
                    camera.matrix = np.array(params["matrix"]).astype(float)
                    camera.distortions = np.array(params["distortions"]).astype(float)
                    camera.grid_count = params["grid_count"]
            # except:
            #     logger.info("Unable to connect... camera may be in use.")

        with ThreadPoolExecutor() as executor:
            for key, params in self.dict.items():
                if key.startswith("cam"):
                    logger.info(f"Beginning to load {key} with params {params}")
                    executor.submit(add_preconfigured_cam, params)
        return cameras


    def save_point_estimates(self, point_estimates:PointEstimates):
        logger.info("Saving point estimates to config...")

        temp_data = asdict(point_estimates)
        for key, params in temp_data.items():
            temp_data[key] = params.tolist()

        self.dict["point_estimates"] = temp_data

        self.update_toml()


    def get_live_stream_pool(self, tracker_enum:TrackerEnum = None):
        streams = {}
        cameras = self.get_cameras()

        if tracker_enum is not None:  
            tracker = tracker_enum.value()
        else:
            tracker = None

        for port, cam in cameras.items():
            logger.info(f"Adding stream associated with camera {port}")
            stream = LiveStream(cam, tracker=tracker)
            stream.change_resolution(cam.size)
            streams[port] = stream
        return streams            
                    
                    
if __name__ == "__main__":
    from pyxy3d import __root__
    
    session_path = Path(__root__,"dev", "sample_sessions", "real_time")
    config = Configurator(session_path)

    

#%%