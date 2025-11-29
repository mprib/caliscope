# more attributes could certainly be tested in here,but at least this gives some basic sense of if things
# are working....
import logging
import shutil
from pathlib import Path

import numpy as np

from caliscope import __root__
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents_to_clean_dest

logger = logging.getLogger(__name__)


def point_estimates_are_equal(pe1: PointEstimates, pe2: PointEstimates) -> bool:
    return (
        np.array_equal(pe1.sync_indices, pe2.sync_indices)
        and np.array_equal(pe1.camera_indices, pe2.camera_indices)
        and np.array_equal(pe1.point_id, pe2.point_id)
        and np.array_equal(pe1.img, pe2.img)
        and np.array_equal(pe1.obj_indices, pe2.obj_indices)
        and np.array_equal(pe1.obj, pe2.obj)
    )


def test_configurator(tmp_path: Path):
    # provided with a path, load toml or create a default toml.
    dev_toml_path = Path(__root__, "tests", "sessions", "post_optimization")

    copy_contents_to_clean_dest(dev_toml_path, tmp_path)

    config = Configurator(tmp_path)

    # load camera array
    camera_array = config.get_camera_array()
    assert isinstance(camera_array, CameraArray)

    config.save_camera_array(camera_array)

    config2 = Configurator(tmp_path)
    camera_array2 = config2.get_camera_array()

    # make sure that the rodrigues conversion isn't messing with anything...
    np.testing.assert_array_almost_equal(camera_array.cameras[0].rotation, camera_array2.cameras[0].rotation, decimal=9)

    # load charuco
    charuco = config.get_charuco()
    assert isinstance(charuco, Charuco)
    assert charuco.columns == 4

    # make a change to charuco and save it
    charuco.columns = 12
    config.save_charuco(charuco)

    # confirm change gets reflected when reloaded
    new_charuco = config.get_charuco()
    assert new_charuco.columns == 12

    # load point estimates
    logger.info("Getting point estimates from config...")
    point_estimates = config.load_point_estimates_from_toml()
    assert isinstance(point_estimates, PointEstimates)

    # delete point estimates data
    config.point_estimates_toml_path.unlink()
    assert not config.point_estimates_toml_path.exists()

    # save point estimates stored in memory
    config.save_point_estimates(point_estimates)

    # confirm it exists
    assert config.point_estimates_toml_path.exists()
    config.refresh_point_estimates_from_toml()

    # create new point estimates with newly saved data
    point_estimates_reloaded = config.load_point_estimates_from_toml()
    assert isinstance(point_estimates_reloaded, PointEstimates)

    assert point_estimates_are_equal(point_estimates, point_estimates_reloaded)


def test_new_cameras(tmp_path: Path):
    """
    With the switch from toml to rtoml, differences in saving `None` values is resulting
    in an inability to load new partially calibrated cameras.

    This test ensures that newly created cameras can be stored and reloaded via config
    and will remain the same.
    """

    config = Configurator(tmp_path)

    cam_1 = CameraData(port=1, size=[1280, 720])
    cam_2 = CameraData(port=2, size=[1280, 720])

    cameras = {1: cam_1, 2: cam_2}
    camera_array = CameraArray(cameras)
    config.save_camera_array(camera_array)
    print(camera_array.cameras)

    # with camera array saved by configurator, there are now many "null" values
    # populated in the toml file. Need to make sure that these are loaded correctly
    config_copy = Configurator(tmp_path)
    camera_array_copy = config_copy.get_camera_array()
    print(camera_array_copy.cameras)

    assert camera_array.cameras[1] == camera_array_copy.cameras[1]
    assert camera_array.cameras[2] == camera_array_copy.cameras[2]


if __name__ == "__main__":
    # test_configurator()
    blank_workspace = Path(__file__).parent / "debug"

    if blank_workspace.exists():
        shutil.rmtree(blank_workspace)
    blank_workspace.mkdir(exist_ok=False)

    test_new_cameras(blank_workspace)
