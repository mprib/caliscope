# The purpose of this module is to create a class that will operate as an 
# interface for a given camera. It will  store data related to
# a variety of camera specific variables and allow setting of these variables:
#
# - source
#   - an integer for a live camera or a path to a video file
# - nickname
# - cv2.VideoCapture object based on source
# - Default Resolution
# - list of possible resolutions
# - exposure
# - intrinsic camera properties
#   - camera matrix
#   - distortion parameters
#
# New camera configurations 
#%%
import cv2
from threading import Thread

# from cameras.get_frame_size import get_nearest_resolution


TEST_FRAME_COUNT = 3
MAX_RESOLUTION_CHECK = 10000

class CameraManager(object):

# https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
# see above for constants used to access properties

    def __init__(self, src):


        # check if source has a data feed before proceeding
        test_capture = cv2.VideoCapture(src)
        for _ in range(0, TEST_FRAME_COUNT):
            success, frame = test_capture.read()

        if success:
            self.src = src
            self.capture = test_capture
            self.active_port = True
            # self._exposure = -5

            self.set_exposure()
            self.set_default_resolution()
            self.set_possible_resolutions()
            
        else:
            self.src = src
            self.capture = None
            self.active_port = False
            raise Exception(f"No input from source {src}")       

    @property
    def exposure(self):
        return self._exposure
        # print("Getting exposure")
        # I'm not convinced that the capture device is sending back correct
        # exposure information, so I will just track independently
    
    @exposure.setter
    def exposure(self, value):
        # print("Setting Exposure")
        self.capture.set(cv2.CAP_PROP_EXPOSURE, value)
        self._exposure = value

    @property
    def width(self):
        # print("Getting width")
        return self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    @width.setter
    def width(self, value):
        # print("Setting width")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, value)


    @property
    def height(self):
        return self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @height.setter
    def height(self, value):
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, value)

    @property
    def resolution(self):
        # print("Getting Resolution")
        return (self.width, self.height)

    @resolution.setter
    def resolution(self, value):
        self.width = value[0]
        self.height = value[1]


    def show_me_worker(self, win_name=None): 
        if not win_name:
            win_name = f"'q' to quit video {self.src}"
        while True:
            success, frame = self.capture.read()
            cv2.imshow(win_name, frame)

            if cv2.waitKey(1) ==ord('q'):
                cv2.destroyWindow(win_name)
                break

    def show_me(self, win_name=None):
        thread = Thread(target=self.show_me_worker, args= (win_name, ), daemon=True)
        thread.start()

    def set_default_resolution(self):
        self.default_resolution = self.resolution

    def set_exposure(self):
        self._exposure = self.capture.get(cv2.CAP_PROP_EXPOSURE)

    def get_nearest_resolution(self, test_width):
        """

        """
        # reminder on implementation: calling property getter of width
        # introduces bug because 'old_width' property getter called at end
        print("Getting nearest resolution")
        old_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, test_width)
        resolution = self.resolution
        # print(resolution)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, old_width)
        # print("Second time around" + str(resolution))
        return resolution

    def set_possible_resolutions(self):
        min_res = self.get_nearest_resolution(0)
        max_res = self.get_nearest_resolution(MAX_RESOLUTION_CHECK)

        min_width = min_res[0]
        max_width = max_res[0]

        STEPS_TO_CHECK = 10 # this is apparently a very fast process

        step_size = int((max_width-min_width)/STEPS_TO_CHECK) # the size of jump to make before checking on the resolution

        resolutions = {min_res, max_res}

        for test_width in range(int(min_width + step_size), 
                                int(max_width - step_size), 
                                int(step_size)):
            new_res = self.get_nearest_resolution(test_width)
            print(new_res)
            resolutions.add(new_res)
        resolutions = list(resolutions)
        resolutions.sort()
        self.possible_resolutions = resolutions
# %%
cam1 = CameraManager(1)

#%% 
cam1.show_me()
import time
for i in range(-7,0):
    print(i)
    cam1.capture.set(cv2.CAP_PROP_EXPOSURE, i)
    print("Actual Exposure is: " + str(cam1.capture.get(cv2.CAP_PROP_EXPOSURE)))
    time.sleep(2)
# cam1.show_me()
print(f"Cam1 Exposure is: {cam1.exposure}")

cam1.exposure = -4

print(f"Cam1 Exposure is: {cam1.exposure}")
#%%
cam1.get_resolution()

#%%
cam1.get_min_resolution()
#%%
cam1.get_max_resolution()
#%%
cam3 = CameraManager(1)
#%%
cam3.get_resolution()

#%%
cam3.get_min_resolution()
#%%
cam3.get_max_resolution()
#
#%%
cams = []
for i in range(0,4):

    if i != 2:
        print(f"Configuring Camera {i}")
        print("Attempting to Create...")
        cams.append(CameraManager(i))
    

# %%
for cam in cams:
    print(f"Camera at port {cam.src}")
    print(f"Active Port?: {cam.active_port}")
    print(f"Exposure: {cam.exposure}")
    cam.show_me()
# %%
