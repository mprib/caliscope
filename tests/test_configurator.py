# more attributes could certainly be tested in here,but at least this gives some basic sense of if things
# are working....
import caliscope.logger

from pathlib import Path
import numpy as np
import os
import shutil

from caliscope import __root__
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.calibration.charuco import Charuco
from caliscope.calibration.capture_volume.point_estimates import PointEstimates

logger = caliscope.logger.get(__name__)

def point_estimates_are_equal(pe1: PointEstimates, pe2: PointEstimates) -> bool:
    return (
        np.array_equal(pe1.sync_indices, pe2.sync_indices) and
        np.array_equal(pe1.camera_indices, pe2.camera_indices) and
        np.array_equal(pe1.point_id, pe2.point_id) and
        np.array_equal(pe1.img, pe2.img) and
        np.array_equal(pe1.obj_indices, pe2.obj_indices) and
        np.array_equal(pe1.obj, pe2.obj)
    )

def test_configurator():
    # provided with a path, load toml or create a default toml.
    dev_toml_path = Path(__root__, "tests", "sessions", "post_optimization")
    test_delete_path = Path(__root__, "tests", "sessions_copy_delete", "post_optimization")

    copy_contents(dev_toml_path,test_delete_path)

    config = Configurator(test_delete_path)

    # load camera array
    camera_array = config.get_camera_array()
    assert(isinstance(camera_array, CameraArray))

    config.save_camera_array(camera_array)

    config2 = Configurator(test_delete_path)
    camera_array2 = config2.get_camera_array()


    # make sure that the rodrigues conversion isn't messing with anything...
    np.testing.assert_array_almost_equal(camera_array.cameras[0].rotation,camera_array2.cameras[0].rotation, decimal=9)

    # load charuco
    charuco = config.get_charuco()
    assert(isinstance(charuco, Charuco))
    assert(charuco.columns==4)

    # make a change to charuco and save it
    charuco.columns = 12
    config.save_charuco(charuco)

    # confirm change gets reflected when reloaded
    new_charuco = config.get_charuco()
    assert(new_charuco.columns==12)

    
    # load point estimates
    logger.info("Getting point estimates from config...")
    point_estimates = config.get_point_estimates()
    assert(type(point_estimates)==PointEstimates)

    # delete point estimates data
    config.point_estimates_toml_path.unlink()
    assert not config.point_estimates_toml_path.exists()
     
    # save point estimates stored in memory 
    config.save_point_estimates(point_estimates)
    
    # confirm it exists 
    assert config.point_estimates_toml_path.exists()
    config.refresh_point_estimates_from_toml()
    
    # create new point estimates with newly saved data
    point_estimates_reloaded = config.get_point_estimates()
    assert(type(point_estimates_reloaded)==PointEstimates)

    assert point_estimates_are_equal(point_estimates, point_estimates_reloaded)

def remove_all_files_and_folders(directory_path):
    for item in directory_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def test_new_cameras():
    """
    With the switch from toml to rtoml, differences in saving `None` values is resulting
    in an inability to load new partially calibrated cameras. 
    
    This test ensures that newly created cameras can be stored and reloaded via config
    and will remain the same.   
    """
    
    blank_workspace = Path(__root__, "tests", "sessions_copy_delete", "blank_workspace")
    if blank_workspace.exists():
        shutil.rmtree(blank_workspace)
    blank_workspace.mkdir(exist_ok=False)
    
    config = Configurator(blank_workspace)

    # note that it appears rtoml will store tuples as lists,
    # so just keep everything a list for comparison purposes
    cam_1 = CameraData(port=1, size=[1280,720])
    cam_2 = CameraData(port=2, size=[1280,720])
                                     
    cameras = {1:cam_1, 2:cam_2}     
    camera_array = CameraArray(cameras)
    config.save_camera_array(camera_array)
    print(camera_array.cameras)
   
    # with camera array saved by configurator, there are now many "null" values 
    # populated in the toml file. Need to make sure that these are loaded correctly
    
    config_copy = Configurator(blank_workspace)
    camera_array_copy = config_copy.get_camera_array()
    print(camera_array_copy.cameras)
    
    assert(camera_array.cameras[1] == camera_array_copy.cameras[1])    
    assert(camera_array.cameras[2] == camera_array_copy.cameras[2])    
    
if __name__ == "__main__":
    # test_configurator()
    test_new_cameras()
    
