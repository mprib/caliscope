
#%% 
from pathlib import Path
from queue import Queue
from time import sleep
from PySide6.QtWidgets import QApplication

import cv2
from pyxy3d import __root__
from pyxy3d.helper import copy_contents
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.interface import FramePacket
from pyxy3d.controller import Controller, read_video_properties
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
def test_controller_load_camera_and_stream():
    """
    Note that in this test the copied workspace config does not have camera data 
    in it, nor mp4s set up for intrinsic calibration (these are in extrinsic).
    
    This is done to make sure it is testing out setting up intrinsic source and config info from imported mp4
    
    """
    app = QApplication()  # must exist prior to QPixels which are downstream when controller is created
    original_workspace = Path(__root__, "tests", "sessions", "prerecorded_calibration")
    workspace = Path( __root__, "tests", "sessions_copy_delete", "prerecorded_calibration")
    copy_contents(original_workspace, workspace)

    controller = Controller(workspace) 
    
    source_0 = Path(workspace,"calibration","extrinsic", "port_0.mp4")
    source_1 = Path(workspace,"calibration","extrinsic", "port_1.mp4")
    controller.add_camera_from_source(source_0)
    controller.add_camera_from_source(source_1)

    assert(len(controller.all_camera_data) ==2)    
    # controller will load in streams used for intrinsic calibration
    controller.load_intrinsic_streams()    
    assert(len(controller.intrinsic_streams) ==2)    

    controller.play_stream(0)
    sleep(.1)
    controller.pause_stream(0)
    controller.stream_jump_to(0,10)
    controller.end_stream(0)
    app.quit()
     
def test_video_property_reader():

    test_source = Path(__root__, "tests", "sessions", "prerecorded_calibration","calibration", "extrinsic", "port_1.mp4")
    logger.info(f"Testing with source file: {test_source}")
    assert(test_source.exists())
    source_properties = read_video_properties(source_path=test_source)
    assert(source_properties["frame_count"]==48)    
    assert(source_properties["fps"]==6.0)    
    assert(source_properties["size"]==(1280,720))


if __name__ == "__main__":
    test_video_property_reader()
    test_controller_load_camera_and_stream()