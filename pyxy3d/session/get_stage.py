"""
A set of functions which are primarily serving the function `get_stage` which will
determine where the user is in the workflow so that the GUI can update accordingly.
The session stage is expressed as an Enum 

"""

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)
from enum import Enum, auto
from itertools import combinations
from pyxy3d.session.session import Session
from pyxy3d.cameras.camera_array import CameraArray

class CameraStage(Enum):
    NO_CAMERAS = auto()
    UNCALIBRATED_CAMERAS = auto()
    INTRINSICS_IN_PROCESES = auto()
    INTRINSICS_ESTIMATED = auto()
    EXTRINSICS_ESTIMATED = auto()
    ORIGIN_SET = auto()
    # RECORDINGS_SAVED = auto()


########################## STAGE ASSOCIATED METHODS #################################
def get_camera_stage(session:Session):
    stage = None
    if connected_camera_count(session) == 0:
        stage = CameraStage.NO_CAMERAS

    elif calibrated_camera_count(session) < connected_camera_count(session):
        stage = CameraStage.UNCALIBRATED_CAMERAS

    elif (
        connected_camera_count(session) > 0
        and calibrated_camera_count(session) == connected_camera_count(session)
    ):
        stage = CameraStage.INTRINSICS_ESTIMATED


    if hasattr(session, "camera_array"):
        if extrinsics_calibrated(session.camera_array.cameras):
            stage = CameraStage.EXTRINSICS_ESTIMATED

    logger.info(f"Current stage of session is {stage}")
    return stage


def connected_camera_count(session:Session):
    """Used to keep track of where the user is in the calibration process"""
    return len(session.cameras)

def calibrated_camera_count(session:Session):
    """Used to keep track of where the user is in the calibration process"""
    count = 0
    for key in session.config.dict.keys():
        if key.startswith("cam"):
            if "error" in session.config.dict[key].keys():
                if session.config.dict[key]["error"] is not None:
                    count += 1
    return count

def extrinsics_calibrated(cameras: dict) -> bool:
    """
    identify the calibration stage of the dictionary of cameras
    """

    #assume true and prove otherwise
    has_extrinsics = True
    for port, camera in cameras.items():
        if camera.ignore == False and (
            camera.rotation is None or camera.translation is None
        ):
            has_extrinsics = False
            logger.info(
                f"Camera array is fully calibrated because camera {port} lacks extrinsics"
            )
            logger.info(f"{port} Rotation: {camera.rotation}")
            logger.info(f"{port} Translation: {camera.translation}")
            logger.info(f"{port} Matrix: {camera.matrix}")

    return has_extrinsics