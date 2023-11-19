# %%

import pyxy3d.logger

from pathlib import Path
from datetime import datetime
from os.path import exists
import numpy as np
import toml
from dataclasses import asdict
import cv2

from pyxy3d.calibration.charuco import Charuco
from pyxy3d.cameras.camera import Camera
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from concurrent.futures import ThreadPoolExecutor

logger = pyxy3d.logger.get(__name__)


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
        else:
            logger.info(
                "No existing config.toml found; creating starter file with charuco"
            )

            self.dict = toml.loads("")
            self.dict["CreationDate"] = datetime.now()
            self.update_config_toml()

            # default values enforced below
            charuco = Charuco(4, 5, 11, 8.5, square_size_overide_cm=5.4)
            self.save_charuco(charuco)
            self.save_fps_recording(30)
            self.save_fps_extrinsic_calibration(6)
            self.save_fps_intrinsic_calibration(6)
            self.save_extrinsic_wait_time(0.5)
            self.save_intrinsic_wait_time(0.5)

        if exists(self.point_estimates_toml_path):
            self.refresh_point_estimates_from_toml()

    def get_intrinsic_wait_time(self):
        return self.dict["intrinsic_wait_time"]

    def get_extrinsic_wait_time(self):
        return self.dict["extrinsic_wait_time"]

    def get_fps_recording(self):
        return self.dict["fps_recording"]

    def get_fps_extrinsic_calibration(self):
        return self.dict["fps_extrinsic_calibration"]

    def get_fps_intrinsic_calibration(self):
        return self.dict["fps_intrinsic_calibration"]

    def save_intrinsic_wait_time(self, time_sec: float):
        logger.info(f"Updating Intrinsic Calibration Wait to to {time_sec}")
        self.dict["intrinsic_wait_time"] = time_sec
        self.update_config_toml()

    def save_extrinsic_wait_time(self, time_sec: float):
        logger.info(f"Updating Extrinsic Calibration Wait to to {time_sec}")
        self.dict["extrinsic_wait_time"] = time_sec
        self.update_config_toml()

    def save_fps_recording(self, fps: int):
        logger.info(f"Updating Recording fps to {fps}")
        self.dict["fps_recording"] = fps
        self.update_config_toml()

    def save_fps_extrinsic_calibration(self, fps: int):
        logger.info(f"Updating Extrinsic calibration fps to {fps}")
        self.dict["fps_extrinsic_calibration"] = fps
        self.update_config_toml()

    def save_fps_intrinsic_calibration(self, fps: int):
        logger.info(f"Updating Intrinsic calibration fps to {fps}")
        self.dict["fps_intrinsic_calibration"] = fps
        self.update_config_toml()

    def refresh_config_from_toml(self):
        logger.info("Populating config dictionary with config.toml data")
        # with open(self.config_toml_path, "r") as f:
        self.dict = toml.load(self.config_toml_path)

    def refresh_point_estimates_from_toml(self):
        logger.info("Populating config dictionary with config.toml data")
        # with open(self.config_toml_path, "r") as f:
        self.dict["point_estimates"] = toml.load(self.point_estimates_toml_path)

    def update_config_toml(self):
        # alphabetize by key to maintain standardized layout
        sorted_dict = {key: value for key, value in sorted(self.dict.items())}
        self.dict = sorted_dict

        dict_wo_point_estimates = {
            key: value for key, value in self.dict.items() if key != "point_estimates"
        }
        with open(self.config_toml_path, "w") as f:
            toml.dump(dict_wo_point_estimates, f)

    def save_capture_volume(self, capture_volume: CaptureVolume):
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
        self.update_config_toml()

    def get_camera_from_source(self, port:int)->CameraData:
        target_mp4_path = Path(self.workspace_path, "calibration", "intrinsic", f"port_{port}.mp4")
        video_properties = read_video_properties(target_mp4_path)
        size = video_properties["size"]
        new_cam_data = CameraData(
            port=port,
            size=size,
        )
        return new_cam_data
    
    def get_all_source_camera_ports(self)-> list:
        target_mp4_dir = Path(self.workspace_path, "calibration", "intrinsic")
        ports = []
        for file in target_mp4_dir.iterdir():
            if file.stem[0:5] == "port_":
                port = file.stem.split("_")[1]
                ports.append(int(port))

        return ports

    def get_configured_camera_data(self) -> CameraArray:
        """
        Load camera array directly from config file. The results of capture volume
        optimization and origin transformation will be reflected in this array
        which can then be the basis for future 3d point estimation
        """
        all_camera_data = {}
        for key, params in self.dict.items():
            if key.startswith("cam"):
                if not params["ignore"]:
                    port = params["port"]

                    if "error" in params.keys():  # intrinsics have been calculated
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
                    ):  # Extrinsics have been calculated
                        translation = np.array(params["translation"])
                        rotation = np.array(params["rotation"])

                        if rotation.shape == (
                            3,
                        ):  # camera rotation is stored as a matrix
                            rotation = cv2.Rodrigues(rotation)[0]

                    else:
                        translation = None
                        rotation = None

                    # if "original_intrinsic_source" in params.keys():
                    #     original_intrinsic_source = params["original_intrinsic_source"]
                    # else:
                    #     original_intrinsic_source = None

                    # if "original_extrinsic_source" in params.keys():
                    #     original_extrinsic_source = params["original_extrinsic_source"]
                    # else:
                    #     original_extrinsic_source = None

                    logger.info(f"Adding camera {port} to calibrated camera array...")
                    cam_data = CameraData(
                        port=port,
                        size=params["size"],
                        rotation_count=params["rotation_count"],
                        error=error,
                        matrix=matrix,
                        distortions=distortions,
                        # exposure=params["exposure"],
                        grid_count=grid_count,
                        ignore=params["ignore"],
                        # verified_resolutions=params["verified_resolutions"],
                        translation=translation,
                        rotation=rotation,
                        # original_intrinsic_source=original_intrinsic_source,
                        # original_extrinsic_source=original_extrinsic_source
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
        # if self.last_point_estimates_save_time > self.last_point_estimates_load_time:
        #     self.dict["point_estimates"] = toml.load(self.point_estimates_toml_path)
        #     self.last_point_estimates_load_time = time()

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

        if camera.rotation is not None:
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
            "ignore": camera.ignore,
            "verified_resolutions": camera.verified_resolutions,
        }

        self.dict["cam_" + str(camera.port)] = params
        self.update_config_toml()

    def save_camera_array(self, camera_array: CameraArray):
        logger.info("Saving camera array....")
        # for port, camera_data in camera_array.cameras.items():
        #     camera_data = camera_array.cameras[port]
        #     self.save_camera(camera_data)
        self.save_all_camera_data(camera_array.cameras)

    def save_all_camera_data(self, all_camera_data: dict):
        logger.info("Saving all camera data")
        for port, camera_data in all_camera_data.items():
            camera_data = all_camera_data[port]
            self.save_camera(camera_data)

    def get_cameras(self) -> dict[Camera]:
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

                if "error" in params.keys():  # then intrinsic params available
                    # if calibration done, then populate those as well
                    logger.info(
                        f"Adding in preconfigured intrinsic params at port {port}"
                    )
                    logger.info(f"Camera RMSE error for port {port}: {params['error']}")
                    camera.error = params["error"]
                    camera.matrix = np.array(params["matrix"]).astype(float)
                    camera.distortions = np.array(params["distortions"]).astype(float)
                    camera.grid_count = params["grid_count"]

                if "rotation" in params.keys():  # then extrinsic params available
                    logger.info(
                        f"Adding in preconfigured extrinsic params at port {port}"
                    )
                    camera.rotation = cv2.Rodrigues(np.array(params["rotation"]))[0]
                    camera.translation = np.array(params["translation"])

        with ThreadPoolExecutor() as executor:
            for key, params in self.dict.items():
                if key.startswith("cam"):
                    logger.info(f"Beginning to load {key} with params {params}")
                    executor.submit(add_preconfigured_cam, params)
        return cameras

    def save_point_estimates(self, point_estimates: PointEstimates):
        logger.info("Saving point estimates to toml...")

        temp_data = asdict(point_estimates)

        for key, params in temp_data.items():
            temp_data[key] = params.tolist()

        self.dict["point_estimates"] = temp_data

        with open(self.point_estimates_toml_path, "w") as f:
            toml.dump(self.dict["point_estimates"], f)
        # self.update_config_toml()

def read_video_properties(source_path: Path) -> dict:
    # Dictionary to hold video properties
    properties = {}

    # Open the video file
    video = cv2.VideoCapture(str(source_path))
    logger.info(f"Attempting to open video file: {source_path}")

    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Could not open the video file: {source_path}")

    # Extract video properties
    properties["frame_count"] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    properties["fps"] = video.get(cv2.CAP_PROP_FPS)
    properties["width"] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties["height"] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    properties["size"] = (properties["width"], properties["height"])

    # Release the video capture object
    video.release()

    return properties

if __name__ == "__main__":
    import toml
    from pyxy3d import __app_dir__

    app_settings = toml.load(Path(__app_dir__, "settings.toml"))
    recent_projects: list = app_settings["recent_projects"]

    recent_project_count = len(recent_projects)
    session_path = Path(recent_projects[recent_project_count - 1])

    config = Configurator(session_path)

# %%
