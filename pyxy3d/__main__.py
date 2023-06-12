import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
import os
from PyQt6.QtWidgets import QApplication
from pathlib import Path

from pyxy3d.gui.calibration_widget import launch_extrinsic_calibration_widget
from pyxy3d.gui.recording_widget import launch_recording_widget
from pyxy3d.gui.main_widget import launch_main

def CLI_parser():
    if len(sys.argv) == 1:
        # logger.warn("No argument supplied")
        # logger.warn(
        #     "Please provide specific function to perform: calibrate, record, process"
        # )
        launch_main()

    if len(sys.argv) == 2:
        session_path = Path(os.getcwd())
        launch_widget = sys.argv[1]

        if launch_widget in ["calibrate", "cal", "-c"]:
            launch_extrinsic_calibration_widget(session_path)

        if launch_widget in ["record", "rec", "-r"]:
            launch_recording_widget(session_path)

