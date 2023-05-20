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

class Stage(Enum):
    NO_CAMERAS = auto()
    UNCALIBRATED_CAMERAS = auto()
    MONOCALIBRATED_CAMERAS = auto()
    OMNICALIBRATION_IN_PROCESS = auto()
    OMNICALIBRATION_DONE = auto()
    ORIGIN_SET = auto()


########################## STAGE ASSOCIATED METHODS #################################
def get_stage(session:Session):
    stage = None
    if connected_camera_count(session) == 0:
        stage = Stage.NO_CAMERAS

    elif calibrated_camera_count(session) < connected_camera_count(session):
        stage = Stage.UNCALIBRATED_CAMERAS

    elif (
        connected_camera_count(session) > 0
        and calibrated_camera_count(session) == connected_camera_count(session)
    ):
        stage = Stage.MONOCALIBRATED_CAMERAS

    elif len(calibrated_camera_pairs(session)) == len(camera_pairs(session)):
        stage = Stage.OMNICALIBRATION_DONE

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

def camera_pairs(session:Session):
    """Used to keep track of where the user is in the calibration process"""
    ports = [key for key in session.cameras.keys()]
    pairs = [pair for pair in combinations(ports, 2)]
    sorted_ports = [
        (min(pair), max(pair)) for pair in pairs
    ]  # sort as in (b,a) --> (a,b)
    sorted_ports = sorted(
        sorted_ports
    )  # sort as in [(b,c), (a,b)] --> [(a,b), (b,c)]
    return sorted_ports

def calibrated_camera_pairs(session:Session):
    """Used to keep track of where the user is in the calibration process"""
    calibrated_pairs = []
    for key in session.config.dict.keys():
        if key.startswith("stereo"):
            portA, portB = key.split("_")[1:3]
            calibrated_pairs.append((int(portA), int(portB)))
    calibrated_pairs = sorted(
        calibrated_pairs
    )  # sort as in [(b,c), (a,b)] --> [(a,b), (b,c)]
    return calibrated_pairs

