# A place for me to sort out the details of this current refactor.
# I want to be able to construct all needed objects from a config file,
# as well as save out modified objects to that config file.

# This will turn into a test of the configurator... Not really looking to 
# perform the calibration. Just load everything, then save everything out, 
# to a different file and make sure stuff is the same.
#%%
from pathlib import Path

from pyxy3d import __root__
from pyxy3d.configurator import Configurator
from pyxy3d.helper import copy_contents
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.charuco import Charuco

# provided with a path, load toml or create a default toml.
dev_toml_path = Path(__root__, "tests", "sessions", "low_res_laptop")
test_delete_path = Path(__root__, "tests", "sessions_copy_delete", "low_res_laptop")

copy_contents(dev_toml_path,test_delete_path)

config = Configurator(test_delete_path)

# check if config has cameras....needed for session to determine if it should find them

# SIDE NOTE HERE: session.find_cameras could just be a standalone helper function.
# might as well hold it in config for the time being...

# load cameras
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
assert(charuco.columns==12)


# load streams *from cameras* (these do not require loading from config)
# note: cannot be tested as part of pytest...requires actual camera connection
live_streams = config.get_live_stream_pool()

# load camera, make a change, save it, reload it, confirm
# change is reflected
cameras = config.get_cameras()



#%%