import logging
from datetime import datetime
from enum import Enum
from os.path import exists
from pathlib import Path

import rtoml

from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope import persistence

logger = logging.getLogger(__name__)


class ProjectSettings(Enum):
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
    Central I/O interface for project state.
    Loads/saves camera arrays, point data, and calibration results
    from the project directory structure.
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = workspace_path
        self.config_toml_path = Path(self.workspace_path, "config.toml")
        self.point_estimates_toml_path = Path(self.workspace_path, "point_estimates.toml")
        self.camera_array_path = Path(self.workspace_path, "camera_array.toml")

        if exists(self.config_toml_path):
            self.refresh_config_from_toml()
            # this check only included for interfacing with historical tests...
            # if underlying tests data updated, this should be removed
            if "camera_count" not in self.dict.keys():
                self.dict["camera_count"] = 0
        else:
            logger.info("No existing config.toml found; creating starter file with charuco")
            self.dict = rtoml.loads("")
            self.dict[ProjectSettings.creation_date.value] = datetime.now()
            self.dict[ProjectSettings.camera_count.value] = 0
            self.dict[ProjectSettings.save_tracked_points_video.value] = True
            self.dict[ProjectSettings.fps_sync_stream_processing.value] = 100
            self.update_config_toml()

            # default values enforced below
            charuco = Charuco(4, 5, 11, 8.5, square_size_overide_cm=5.4)
            self.save_charuco(charuco)

            # Create empty camera array for new projects
            empty_array = CameraArray({})
            self.save_camera_array(empty_array)

    def save_camera_count(self, count):
        self.camera_count = count
        self.dict[ProjectSettings.camera_count.value] = count
        self.update_config_toml()

    def get_camera_count(self):
        return self.dict[ProjectSettings.camera_count.value]

    def get_save_tracked_points(self):
        if ProjectSettings.save_tracked_points_video.value not in self.dict.keys():
            return True
        else:
            return self.dict[ProjectSettings.save_tracked_points_video.value]

    def get_fps_sync_stream_processing(self):
        if ProjectSettings.fps_sync_stream_processing.value not in self.dict.keys():
            return 100
        else:
            return self.dict[ProjectSettings.fps_sync_stream_processing.value]

    def refresh_config_from_toml(self):
        """Load project settings from dedicated file."""
        logger.info("Loading project settings from project_settings.toml")
        settings_path = self.workspace_path / "project_settings.toml"
        try:
            self.dict = persistence.load_project_settings(settings_path)
        except persistence.PersistenceError as e:
            logger.error(f"Failed to load project settings: {e}")
            # Initialize with empty dict for new projects
            self.dict = {}

    def refresh_point_estimates_from_toml(self):
        logger.info("Populating config dictionary with point_estimates.toml data")
        # with open(self.config_toml_path, "r") as f:
        self.dict["point_estimates"] = rtoml.load(self.point_estimates_toml_path)

    def update_config_toml(self):
        """Save project settings to dedicated file."""
        # Filter out point_estimates if it's in dict (legacy compatibility)
        dict_wo_point_estimates = {k: v for k, v in self.dict.items() if k != "point_estimates"}

        settings_path = self.workspace_path / "project_settings.toml"
        try:
            persistence.save_project_settings(dict_wo_point_estimates, settings_path)
        except persistence.PersistenceError as e:
            logger.error(f"Failed to save project settings: {e}")
            raise

    def save_capture_volume(self, capture_volume: CaptureVolume):
        """Delegate to persistence layer for both camera array and metadata."""
        # Save camera array
        self.save_camera_array(capture_volume.camera_array)

        # Save point estimates
        self.save_point_estimates(capture_volume.point_estimates)

        # Save capture volume metadata
        metadata = {
            "stage": capture_volume.stage,
            "origin_sync_index": capture_volume.origin_sync_index,
        }
        metadata_path = self.workspace_path / "capture_volume.toml"
        try:
            persistence.save_capture_volume_metadata(metadata, metadata_path)
        except persistence.PersistenceError as e:
            logger.error(f"Failed to save capture volume metadata: {e}")
            raise

    def get_configured_camera_data(self) -> dict[int, CameraData]:
        """
        Load camera data from dedicated camera_array.toml file.
        """
        try:
            camera_array = persistence.load_camera_array(self.camera_array_path)
            return camera_array.cameras
        except persistence.PersistenceError as e:
            logger.error(f"Failed to load camera array: {e}")
            raise

    def get_camera_array(self) -> CameraArray:
        """
        Load camera array directly from camera_array.toml file.
        """
        try:
            return persistence.load_camera_array(self.camera_array_path)
        except persistence.PersistenceError as e:
            logger.error(f"Failed to load camera array: {e}")
            raise

    def load_point_estimates_from_toml(self) -> PointEstimates:
        """Load point estimates from dedicated file."""
        path = self.workspace_path / "point_estimates.toml"
        try:
            return persistence.load_point_estimates(path)
        except persistence.PersistenceError as e:
            logger.error(f"Failed to load point estimates: {e}")
            raise

    def get_charuco(self) -> Charuco:
        """
        Load Charuco from dedicated charuco.toml file.
        File must exist and be valid - no fallback to legacy config.toml.
        """
        charuco_path = self.workspace_path / "charuco.toml"
        return persistence.load_charuco(charuco_path)

    def save_charuco(self, charuco: Charuco):
        charuco_path = self.workspace_path / "charuco.toml"
        try:
            persistence.save_charuco(charuco, charuco_path)
            logger.info(f"Charuco saved to {charuco_path}")
        except persistence.PersistenceError as e:
            logger.error(f"Failed to save charuco: {e}")
            raise

    def save_camera(self, camera: CameraData):
        """
        Save a single camera by loading the full array, updating it, and saving back.
        Inefficient but maintains API compatibility during transition.
        """
        try:
            # Load existing array (or empty if new project)
            camera_array = persistence.load_camera_array(self.camera_array_path)

            # Update the specific camera
            camera_array.cameras[camera.port] = camera

            # Save back
            persistence.save_camera_array(camera_array, self.camera_array_path)
            logger.info(f"Camera {camera.port} saved to {self.camera_array_path}")
        except persistence.PersistenceError as e:
            logger.error(f"Failed to save camera {camera.port}: {e}")
            raise

    def save_camera_array(self, camera_array: CameraArray):
        """
        Save entire camera array to dedicated file.
        """
        try:
            persistence.save_camera_array(camera_array, self.camera_array_path)
            logger.info(f"Camera array saved to {self.camera_array_path}")
        except persistence.PersistenceError as e:
            logger.error(f"Failed to save camera array: {e}")
            raise

    def save_point_estimates(self, point_estimates: PointEstimates):
        """Save point estimates to dedicated file."""
        path = self.point_estimates_toml_path
        try:
            persistence.save_point_estimates(point_estimates, path)
        except persistence.PersistenceError as e:
            logger.error(f"Failed to save point estimates: {e}")
            raise


if __name__ == "__main__":
    import rtoml

    from caliscope import __app_dir__

    app_settings = rtoml.load(Path(__app_dir__, "settings.toml"))
    recent_projects: list = app_settings["recent_projects"]

    recent_project_count = len(recent_projects)
    session_path = Path(recent_projects[recent_project_count - 1])

    config = Configurator(session_path)

# %%
