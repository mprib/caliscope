"""

This code may not be relevant to anything. I believe that given the fact that 
the camera array is established only after the extrinsic calibration, it is not 
really important to track whether the camera array is calibrated. We know it's 
calibrated. It should be the case that the recording is only possible when there 
is currently a calibrated camera array that is already embedded in the session 
stage. The recorder should store the recorded file.


"""

# %%

import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from pathlib import Path

from pyxy3d.configurator import Configurator
from pyxy3d import __root__
from pyxy3d.cameras.camera_array import CameraArray, CalibrationStage
from pyxy3d.session.get_stage import get_camera_stage, extrinsics_calibrated, CameraStage
from pyxy3d.session.session import Session, SessionMode

uncalibrated_path = Path(__root__, "tests", "reference", "only_intrinsics")
uncalibrated_config = Configurator(uncalibrated_path)
uncalibrated_camera_array = uncalibrated_config.get_camera_array()
# %%
# assume that it is eligible and attempt to prove that wrong


assert not extrinsics_calibrated(uncalibrated_camera_array.cameras)

session = Session(uncalibrated_config)
stage = get_camera_stage(session)
logger.info(f"stage of uncalibrated session is: {stage}")
