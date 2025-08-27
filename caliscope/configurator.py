# %%

from dataclasses import asdict
from datetime import datetime
from enum import Enum
from os.path import exists
from pathlib import Path

import cv2
import numpy as np
import rtoml

import caliscope.logger
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.charuco import Charuco
from caliscope.cameras.camera import Camera
from caliscope.cameras.camera_array import CameraArray, CameraData

logger = caliscope.logger.get(__name__)


class ConfigSettings(Enum):
    """
    Control settings used to manage the processing of data
    """

    creation_date = "creation_date"
    camera_count = "camera_count"
    save_tracked_points_video = "save_tracked_points_video"
    fps_sync_stream_processing = "fps_sync_stream_processing"


# %%


class Configurator:
    """
    A factory to provide pre-configured objects and to save the configuration
    of currently existing objects.
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = workspace_path
        self.config_toml_path = Path(self.workspace_path, "config.toml")
        self.point_estimates_toml_path = Path(self.workspace_path, "point_estimates.toml")

        if exists(self.config_toml_path):
            self.refresh_config_from_toml()
            # this check only included for interfacing with historical tests...
            # if underlying tests data updated, this should be removed
            if "camera_count" not in self.dict.keys():
                self.dict["camera_count"] = 0
        else:
            logger.info("No existing config.toml found; creating starter file with charuco")
            self.dict = rtoml.loads("")
            self.dict[ConfigSettings.creation_date.value] = datetime.now()
            self.dict[ConfigSettings.camera_count.value] = 0
            self.dict[ConfigSettings.save_tracked_points_video.value] = True
            self.dict[ConfigSettings.fps_sync_stream_processing.value] = 100
            self.update_config_toml()

            # default values enforced below
            charuco = Charuco(4, 5, 11, 8.5, square_size_overide_cm=5.4)
            self.save_charuco(charuco)

    def save_camera_count(self, count):
        self.camera_count = count
        self.dict[ConfigSettings.camera_count.value] = count
        self.update_config_toml()

    def get_camera_count(self):
        return self.dict[ConfigSettings.camera_count.value]

    def get_save_tracked_points(self):
        if ConfigSettings.save_tracked_points_video.value not in self.dict.keys():
            return True
        else:
            return self.dict[ConfigSettings.save_tracked_points_video.value]

    def get_fps_sync_stream_processing(self):
        if ConfigSettings.fps_sync_stream_processing.value not in self.dict.keys():
            return 100
        else:
            return self.dict[ConfigSettings.fps_sync_stream_processing.value]

    def refresh_config_from_toml(self):
        logger.info("Populating config dictionary with config.toml data")
        # with open(self.config_toml_path, "r") as f:
        self.dict = rtoml.load(self.config_toml_path)

    def refresh_point_estimates_from_toml(self):
        logger.info("Populating config dictionary with point_estimates.toml data")
        # with open(self.config_toml_path, "r") as f:
        self.dict["point_estimates"] = rtoml.load(self.point_estimates_toml_path)

    def update_config_toml(self):
        # alphabetize by key to maintain standardized layout
        sorted_dict = {key: value for key, value in sorted(self.dict.items())}
        self.dict = sorted_dict

        dict_wo_point_estimates = {key: value for key, value in self.dict.items() if key != "point_estimates"}
        with open(self.config_toml_path, "w") as f:
            rtoml.dump(dict_wo_point_estimates, f)

    def save_capture_volume(self, capture_volume: CaptureVolume):
        # self.point_estimates = self.capture_volume.point_estimates
        # self.camera_array = self.capture_volume.camera_array
        self.save_camera_array(capture_volume.camera_array)
        self.save_point_estimates(capture_volume.point_estimates)

        self.dict["capture_volume"] = {}
        # self["capture_volume"]["RMSE_summary"] = self.capture_volume.rmse
        self.dict["capture_volume"]["stage"] = capture_volume.stage
        self.dict["capture_volume"]["origin_sync_index"] = capture_volume.origin_sync_index
        self.update_config_toml()

    def get_configured_camera_data(self) -> dict[int, CameraData]:
        all_camera_data = {}

        for key, params in self.dict.items():
            if key.startswith("cam_"):
                port = params["port"]

                if (
                    "error" in params.keys() and params["error"] is not None and params["error"] != "null"
                ):  # intrinsics have been calculated
                    error = params["error"]
                    matrix = np.array(params["matrix"])
                    distortions = np.array(params["distortions"])
                    grid_count = params["grid_count"]
                else:
                    error = None
                    matrix = None
                    distortions = None
                    grid_count = None

                if (
                    "translation" in params.keys()
                    and params["translation"] is not None
                    and params["translation"] != "null"
                ):  # Extrinsics have been calculated
                    translation = np.array(params["translation"])
                    rotation = np.array(params["rotation"])

                    if rotation.shape == (3,):  # camera rotation is stored as a matrix
                        rotation = cv2.Rodrigues(rotation)[0]

                else:
                    translation = None
                    rotation = None

                if "ignore" in params.keys():
                    ignore = params["ignore"]
                else:
                    ignore = False

                logger.info(f"Adding camera {port} to calibrated camera array...")
                cam_data = CameraData(
                    port=port,
                    size=params["size"],
                    rotation_count=params["rotation_count"],
                    error=error,
                    matrix=matrix,
                    distortions=distortions,
                    grid_count=grid_count,
                    ignore=ignore,
                    translation=translation,
                    rotation=rotation,
                )

                all_camera_data[port] = cam_data
                logger.info(f"Camera successfully added at port {port}")

        logger.info("Camera data loaded and being passed back to caller")
        return all_camera_data

    def get_camera_array(self) -> CameraArray:
        """
        Load camera array directly from config file. The results of capture volume
        optimization and origin transformation will be reflected in this array
        which can then be the basis for future 3d point estimation
        """
        all_camera_data = self.get_configured_camera_data()
        camera_array = CameraArray(all_camera_data)
        logger.info("Camera array successfully created and being passed back to caller")
        return camera_array

    def get_point_estimates(self) -> PointEstimates:
        # only load point estimates into dictionary if saved more recently than last loaded

        if "point_estimates" not in self.dict.keys():
            self.refresh_point_estimates_from_toml()

        temp_data = self.dict["point_estimates"].copy()
        for key, value in temp_data.items():
            temp_data[key] = np.array(value)

        point_estimates = PointEstimates(**temp_data)

        return point_estimates

    def get_charuco(self) -> Charuco:
        """
        Charuco will always be available as it is created when initializing the config
        """

        logger.info("Loading charuco from config")
        params = self.dict["charuco"]

        # the below is intended to catch people using older configs without this info
        # if you come across this in mid 2025 or later, these may be safe to delete
        logger.info(f"charuco param are: {params}:")
        if "legacy_pattern" not in params.keys():
            params["legacy_pattern"] = False

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
            legacy_pattern=params["legacy_pattern"],
        )

        return charuco

    def save_charuco(self, charuco: Charuco):
        self.dict["charuco"] = charuco.__dict__
        logger.info(f"Saving charuco with params {charuco.__dict__} to config")
        self.update_config_toml()

    def save_camera(self, camera: Camera | CameraData):
        def none_or_list(value):
            # required to make sensible numeric format
            # otherwise toml formats as text
            if value is None:
                return None
            else:
                return value.tolist()

        if camera.rotation is not None and camera.rotation.any():
            # store rotation as 3 parameter rodrigues
            rotation_for_config = cv2.Rodrigues(camera.rotation)[0][:, 0]
            rotation_for_config = rotation_for_config.tolist()
        else:
            rotation_for_config = None

        params = {
            "port": camera.port,
            "size": camera.size,
            "rotation_count": camera.rotation_count,
            "error": camera.error,
            "matrix": none_or_list(camera.matrix),
            "distortions": none_or_list(camera.distortions),
            "translation": none_or_list(camera.translation),
            "rotation": rotation_for_config,
            "exposure": camera.exposure,
            "grid_count": camera.grid_count,
        }

        self.dict["cam_" + str(camera.port)] = params
        self.update_config_toml()

    def save_camera_array(self, camera_array: CameraArray):
        logger.info("Saving camera array....")
        for port, camera_data in camera_array.cameras.items():
            self.save_camera(camera_data)

    def save_point_estimates(self, point_estimates: PointEstimates):
        logger.info("Saving point estimates to toml...")

        temp_data = asdict(point_estimates)

        for key, params in temp_data.items():
            temp_data[key] = params.tolist()

        self.dict["point_estimates"] = temp_data

        with open(self.point_estimates_toml_path, "w") as f:
            rtoml.dump(self.dict["point_estimates"], f)


if __name__ == "__main__":
    import rtoml

    from caliscope import __app_dir__

    app_settings = rtoml.load(Path(__app_dir__, "settings.toml"))
    recent_projects: list = app_settings["recent_projects"]

    recent_project_count = len(recent_projects)
    session_path = Path(recent_projects[recent_project_count - 1])

    config = Configurator(session_path)

# %%
