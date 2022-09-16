# %%

from operator import inv
from camera import Camera
from charuco import Charuco

charuco = Charuco(4,5,11,8.5)


feeds = {0: "Cam_1"}

vid_file = 'videos\charuco.mkv'

# %%
active_camera = Camera(0, "Cam_1")
active_camera.collect_calibration_corners(
    board_threshold=0.5,
    self.charuco = charuco, 
    charuco_inverted=True,
    time_between_cal=.5) # seconds that must pass before new corners are stored




# %%

active_camera.close()
# %%
