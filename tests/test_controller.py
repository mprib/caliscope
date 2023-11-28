
#%% 
from pathlib import Path
from time import sleep
from PySide6.QtWidgets import QApplication

from pyxy3d import __root__
from pyxy3d.helper import copy_contents
from pyxy3d.controller import Controller
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

    controller.add_camera_from_source(0)
    controller.add_camera_from_source(1)

    assert(len(controller.all_camera_data) ==2)    
    # controller will load in streams used for intrinsic calibration
    controller.load_intrinsic_streams()    
    assert(len(controller.intrinsic_streams) ==2)    
    ports = controller.config.get_all_source_camera_ports()
    assert(ports == [0,1,2,3]) # there are 4 mp4 files in the intrinsic folder

    for port in ports:
        if port not in controller.all_camera_data:
            controller.add_camera_from_source(port)

    assert(list(controller.all_camera_data.keys()) == [0,1,2,3])
        
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
    original_workspace = Path(__root__, "tests", "sessions", "prerecorded_calibration")
    workspace = Path( __root__, "tests", "sessions_copy_delete", "prerecorded_calibration")
    copy_contents(original_workspace, workspace)


if __name__ == "__main__":
    # test_controller_load_camera_and_stream()
    test_extrinsic_calibration()
# %%
