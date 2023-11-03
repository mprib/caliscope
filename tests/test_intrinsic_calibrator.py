import pyxy3d.logger
from pathlib import Path
from pyxy3d import __root__
from pyxy3d.helper import copy_contents


logger = pyxy3d.logger.get(__name__)
def test_intrinsic_calibrator():
    
    # use a general video file with a charuco for convenience
    original_data_path= Path(__root__, "tests", "sessions", "4_cam_recording")
    destination_path =Path(__root__, "tests", "sessions_copy_delete", "4_cam_recording")
    copy_contents(original_data_path,destination_path)




if __name__ == "__main__":
    test_intrinsic_calibrator()
    
