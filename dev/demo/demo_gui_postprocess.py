"""
working file for development of a post processing widget
"""
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)
from PyQt6.QtWidgets import QApplication
from pyxy3d.gui.post_processing_widget import PostProcessingWidget
import sys
import toml
from time import sleep
from pyxy3d import __root__
from pyxy3d.configurator import Configurator
from pathlib import Path
from pyxy3d import __app_dir__


app_settings = toml.load(Path(__app_dir__, "settings.toml"))
recent_projects:list = app_settings["recent_projects"]

recent_project_count = len(recent_projects)
session_path = Path(recent_projects[recent_project_count-1])
logger.info(f"Launching post processing widget for {session_path}")

config =  Configurator(session_path)

app = QApplication(sys.argv)
window = PostProcessingWidget(config)

# open in a session already so you don't have to go through the menu each time
# window.open_session(config_path)

window.show()

app.exec()