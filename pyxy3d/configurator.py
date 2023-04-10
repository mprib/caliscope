
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from pathlib import Path
from datetime import datetime
from os.path import exists
import numpy as np
import toml
from dataclasses import asdict

from pyxy3d.calibration.charuco import Charuco
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates, load_point_estimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume

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
                self.config = toml.load(self.toml_path)
        else:
            logger.info(
                "No existing config.toml found; creating starter file with charuco"
            )

            self.config = toml.loads("")
            self.config["CreationDate"] = datetime.now()
            with open(self.toml_path, "a") as f:
                toml.dump(self.config, f)


    def update_config(self):
        # alphabetize by key to maintain standardized layout
        sorted_config = {key: value for key, value in sorted(self.config.items())}
        self.config = sorted_config

        with open(self.toml_path, "w") as f:
            toml.dump(self.config, f)
        
    def get_camera_array(self)->CameraArray:
        """
        Load camera array directly from config file. The results of capture volume
        optimization and origin transformation will be reflected in this array
        which can then be the basis for future 3d point estimation
        """
        all_camera_data = {}
        for key, params in self.config.items():
            if key.startswith("cam"):
                if params["translation"] is not None:
                    port = params["port"]
                    size = params["size"]

                    logger.info(f"Adding camera {port} to calibrated camera array...")
                    cam_data = CameraData(
                        port=port,
                        size=params["size"],
                        rotation_count=params["rotation_count"],
                        error=params["error"],
                        matrix=np.array(params["matrix"]),
                        distortions=np.array(params["distortions"]),
                        exposure=params["exposure"],
                        grid_count=params["grid_count"],
                        ignore=params["ignore"],
                        verified_resolutions=params["verified_resolutions"],
                        translation=np.array(params["translation"]),
                        rotation=np.array(params["rotation"]),
                    )

                    all_camera_data[port] = cam_data

        camera_array = CameraArray(all_camera_data)
        return camera_array
    
    
    def get_point_estimates(self)->PointEstimates:
        point_estimates_dict = self.config["point_estimates"]

        for key, value in point_estimates_dict.items():
            point_estimates_dict[key] = np.array(value)

        point_estimates = PointEstimates(**point_estimates_dict)
        return point_estimates
    
    def get_charuco(self)-> Charuco:
        """
        Helper function to load a pre-configured charuco from a config.toml
        """
        params = self.config["charuco"]

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
    
        return charuco

    
    def save_charuco(self, charuco:Charuco):
        self.config["charuco"] = charuco.__dict__
        logger.info(f"Saving charuco with params {charuco.__dict__} to config")
        self.update_config()

    def save_camera(self, camera):
        def none_or_list(value):

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

        self.config["cam_" + str(camera.port)] = params
        self.update_config()

    def save_camera_array(self, camera_array:CameraArray):
        logger.info("Saving camera array....")
        for port, camera_data in camera_array.cameras.items():
            camera_data = camera_array.cameras[port]
            params = {
                "port": camera_data.port,
                "size": camera_data.size,
                "rotation_count": camera_data.rotation_count,
                "error": camera_data.error,
                "matrix": camera_data.matrix.tolist(),
                "distortions": camera_data.distortions.tolist(),
                "exposure": camera_data.exposure,
                "grid_count": camera_data.grid_count,
                "ignore": camera_data.ignore,
                "verified_resolutions": camera_data.verified_resolutions,
                "translation": camera_data.translation.tolist(),
                "rotation": camera_data.rotation.tolist(),
            }

            self.config["cam_" + str(camera_data.port)] = params

        self.update_config()

    def save_point_estimates(self, point_estimates:PointEstimates):
        logger.info("Saving point estimates to config...")

        temp_data = asdict(point_estimates)
        for key, params in temp_data.items():
            temp_data[key] = params.tolist()

        self.config["point_estimates"] = temp_data

        self.update_config()