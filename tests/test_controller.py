
#%% 
from pathlib import Path
from time import sleep
from PySide6.QtWidgets import QApplication

from caliscope import __root__
from caliscope.cameras.camera_array import CameraArray
from caliscope.helper import copy_contents
from caliscope.controller import Controller, read_video_properties
import caliscope.logger


logger = caliscope.logger.get(__name__)

def test_extrinsic_calibration():
    # app = QApplication()  # must exist prior to QPixels which are downstream when controller is created
    original_workspace = Path(__root__, "tests", "sessions", "post_monocal")
    workspace = Path( __root__, "tests", "sessions_copy_delete", "post_monocal")
    copy_contents(original_workspace, workspace)

    controller = Controller(workspace_dir=workspace)
    
    # calibration requires a capture volume object which is composed of both a camera array, 
    # and a set of point estimates
    controller.load_camera_array()

    # want to make sure that no previously stored data is leaking into this test
    for cam in controller.camera_array.cameras.values():
        cam.rotation = None
        cam.translation = None

    assert(not controller.camera_array.all_extrinsics_calibrated())

    # with the charuco points tracked and saved out, the calibration can now proceed
    controller.calibrate_capture_volume()

    while not controller.camera_array.all_extrinsics_calibrated():
        sleep(1)
        logger.info("waiting on camera array to finalize calibration...")

    logger.info(f"New Camera array is {controller.camera_array}")
    assert(controller.camera_array.all_extrinsics_calibrated())

def test_video_property_reader():

    test_source = Path(__root__, "tests", "sessions", "prerecorded_calibration","calibration", "intrinsic", "port_1.mp4")
    logger.info(f"Testing with source file: {test_source}")
    assert(test_source.exists())
    source_properties = read_video_properties(source_path=test_source)
    assert(source_properties["frame_count"]==48)    
    assert(source_properties["fps"]==6.0)    
    assert(source_properties["size"]==(1280,720))

# if __name__ == "__main__":
    # test_extrinsic_calibration()
    # test_video_property_reader()

