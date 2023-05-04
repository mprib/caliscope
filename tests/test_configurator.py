# more attributes could certainly be tested in here,but at least this gives some basic sense of if things
# are working....
from pathlib import Path

from pyxy3d import __root__
from pyxy3d.configurator import Configurator
from pyxy3d.helper import copy_contents
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.charuco import Charuco


def test_configurator():
    # provided with a path, load toml or create a default toml.
    dev_toml_path = Path(__root__, "tests", "sessions", "post_optimization")
    test_delete_path = Path(__root__, "tests", "sessions_copy_delete", "post_optimization")

    copy_contents(dev_toml_path,test_delete_path)

    config = Configurator(test_delete_path)

    # load camera array
    camera_array = config.get_camera_array()
    assert(isinstance(camera_array, CameraArray))

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


if __name__ == "__main__":
    test_configurator()