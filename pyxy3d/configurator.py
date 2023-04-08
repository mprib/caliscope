
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from pathlib import Path
import numpy as np
import toml

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
        self.config_path = Path(self.session_path,"config.toml")
        
        with open(self.config_path, "r") as f:
            self.config = toml.load(self.config_path)

    def update_config(self):
        # alphabetize by key to maintain standardized layout
        sorted_config = {key: value for key, value in sorted(self.config.items())}
        self.config = sorted_config

        with open(self.config_path, "w") as f:
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
    
    
    def load_point_estimates(self)->PointEstimates:
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