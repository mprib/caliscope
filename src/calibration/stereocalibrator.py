 # I feel like I've forgotten how to program. OK. what the heck and I doing here?



from threading import Thread
import cv2
import time
import sys

import numpy as np
# Append main repo to top of path to allow import of backend
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera
from calibration.mono_calibrator import MonoCalibrator
from src.calibration.charuco import Charuco

from src.session import Session





if __name__ == "__main__":

    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    session.load_cameras()
    session.find_additional_cameras() 

    session.load_rtds()
    session.adjust_resolutions()

