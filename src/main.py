# %%

from operator import inv
from camera import Camera
from charuco import Charuco
from stereocamera import StereoCamera

import json


# %%
# Calibrate 2 cameras to get input parameters for stereo calibration

# charuco = Charuco(4,5,11,8.5)

# cam_0 = Camera(0, "cam_0")
# cam_0.collect_calibration_corners(
#     board_threshold=0.5,
#     charuco = charuco, 
#     charuco_inverted=True,
#     time_between_cal=.5) # seconds that must pass before new corners are stored
# cam_0.calibrate()
# cam_0.save_calibration("calibration_params")

# cam_1 = Camera(1, "cam_1")
# cam_1.collect_calibration_corners(
#     board_threshold=0.5,
#     charuco = charuco, 
#     charuco_inverted=True,
#     time_between_cal=.5) # seconds that must pass before new corners are stored
# cam_1.calibrate()
# cam_1.save_calibration("calibration_params")


# %%
# Collect Dual Data For Stereocalibration



# %%
# test out develoment of StereoCamera object
stereocam = StereoCamera("cam_0", "cam_1", "calibration_params")
# %%
