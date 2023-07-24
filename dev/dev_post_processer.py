from pyxy3d.post_processing.post_processor import PostProcessor
import toml
from pathlib import Path
from pyxy3d.configurator import Configurator
from pyxy3d import __app_dir__, __root__
from pyxy3d.trackers.tracker_enum import TrackerEnum

app_settings = toml.load(Path(__app_dir__, "settings.toml"))
recent_projects:list = app_settings["recent_projects"]

recent_project_count = len(recent_projects)
session_path = Path(recent_projects[recent_project_count-1])


config =  Configurator(session_path)

post_processor = PostProcessor(config)

recording_path = Path(session_path, "recording_2")
post_processor.create_xyz(recording_path, TrackerEnum.HOLISTIC_OPENSIM)