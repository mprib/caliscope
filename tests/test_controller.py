
#%% 
from pathlib import Path
from time import sleep
from PySide6.QtWidgets import QApplication

from pyxy3d import __root__
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.helper import copy_contents
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

    controller._add_camera_from_source(0)
    controller._add_camera_from_source(1)

    assert(len(controller.camera_array.cameras) ==2)    
    # controller will load in streams used for intrinsic calibration
    controller.load_intrinsic_stream_manager()    
    assert(len(controller.intrinsic_stream_manager.streams) ==2)    

    # start fresh
    controller.load_camera_array()
    assert(list(controller.camera_array.cameras.keys()) == [0,1,2,3])
    controller.load_intrinsic_stream_manager()
        
    controller.play_intrinsic_stream(0)
    sleep(.1)
    controller.pause_intrinsic_stream(0)
    sleep(.1)
    controller.stream_jump_to(0,10)
    sleep(.1)
    controller.end_stream(0)
    app.quit()


def test_extrinsic_calibration():
    # app = QApplication()  # must exist prior to QPixels which are downstream when controller is created
    original_workspace = Path(__root__, "tests", "sessions", "post_monocal")
    workspace = Path( __root__, "tests", "sessions_copy_delete", "post_monocal")
    copy_contents(original_workspace, workspace)

    controller = Controller(workspace_dir=workspace)
    
    # calibration requires a capture volume object which is composed of both a camera array, 
    # and a set of point estimates
    controller.load_camera_array()
    controller.load_extrinsic_stream_manager()

    # want to make sure that no previously stored data is leaking into this test
    
    for cam in controller.camera_array.cameras.values():
        cam.rotation = None
        cam.translation = None

    assert(not controller.camera_array.all_extrinsics_calibrated())
    controller.process_extrinsic_streams(fps_target=100)

    xy_path = Path(workspace,"calibration", "extrinsic", "CHARUCO", "xy_CHARUCO.csv")


    while not xy_path.exists():
        sleep(1)
        logger.info(f"Waiting on data to populate in {xy_path}")

    # with the charuco points tracked and saved out, the calibration can now proceed
    controller.estimate_extrinsics()

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

if __name__ == "__main__":
    # test_controller_load_camera_and_stream()
    test_extrinsic_calibration()
    # test_video_property_reader()

