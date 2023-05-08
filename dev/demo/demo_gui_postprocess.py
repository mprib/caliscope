"""
working file for development of a post processing widget
"""
from PyQt6.QtWidgets import QApplication
from pyxy3d.gui.post_processing_widget import PostProcessingWidget
import sys
from time import sleep
from pyxy3d import __root__
from pyxy3d.configurator import Configurator
from pathlib import Path

session_path = Path(__root__, "tests" , "sessions_copy_delete", "4_cam_recording")
config =  Configurator(session_path)

app = QApplication(sys.argv)
window = PostProcessingWidget(config)

# open in a session already so you don't have to go through the menu each time
# window.open_session(config_path)

window.show()

app.exec()