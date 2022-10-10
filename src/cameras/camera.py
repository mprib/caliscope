# The purpose of this module is to create a class that will operate as an 
# interface for a given camera. It will  store data related to
# a variety of camera specific variables and allow setting of these variables:
#
# - port
#   - an integer for a live camera 
#   - note this is focused on live feeds...pre-recorded video doesn't need this
# - nickname
# - cv2.VideoCapture object based on port
# - Default Resolution
# - list of possible resolutions
# - exposure
# - intrinsic camera properties (to be set following calibration/on load)
#   - camera matrix
#   - distortion parameters
#
# New camera configurations 
#%%

import queue
import cv2
from threading import Thread
import time

TEST_FRAME_COUNT = 3
MAX_RESOLUTION_CHECK = 10000

class Camera(object):

# https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
# see above for constants used to access properties
    def __init__(self, port):

        # check if source has a data feed before proceeding
        test_capture = cv2.VideoCapture(port)
        for _ in range(0, TEST_FRAME_COUNT):
            success, frame = test_capture.read()

        if success:
            self.port = port
            self.capture = test_capture
            self.active_port = True
            self.is_connected = True
            self.is_rolling = False

            self.set_exposure()
            self.set_default_resolution()
            self.set_possible_resolutions()
            
        else:
            self.port = port
            self.capture = None
            self.active_port = False
            raise Exception(f"No input from source {port}")       

    @property
    def exposure(self):
        return self._exposure
    
    @exposure.setter
    def exposure(self, value):
        """Note that OpenCV appears to change the exposure value, but 
        this is not read back accurately through the getter, so just
        set it manually here after updating"""
        self.capture.set(cv2.CAP_PROP_EXPOSURE, value)
        self._exposure = value

    @property
    def _width(self):
        return self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    @_width.setter
    def _width(self, value):
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, value)

    @property
    def _height(self):
        return self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @_height.setter
    def _height(self, value):
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, value)

    @property
    def resolution(self):
        return (self._width, self._height)

    @resolution.setter
    def resolution(self, value):
        """Currently, this is how the resolution is actually changed"""
        self._width = value[0]
        self._height = value[1]
        

    def set_default_resolution(self):
        """called at initilization before anything has changed"""
        self.default_resolution = self.resolution

    def set_exposure(self):
        """Need an initial value, though it does not appear that updates to 
        exposure reliably read back from """
        self._exposure = self.capture.get(cv2.CAP_PROP_EXPOSURE)

    def get_nearest_resolution(self, test_width):
        """

        """
        # print("Getting nearest resolution")
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

        STEPS_TO_CHECK = 10 # fast to check so cover your bases

        # the size of jump to make before checking on the resolution
        step_size = int((max_width-min_width)/STEPS_TO_CHECK) 

        resolutions = {min_res, max_res}

        for test_width in range(int(min_width + step_size), 
                                int(max_width - step_size), 
                                int(step_size)):
            new_res = self.get_nearest_resolution(test_width)
            # print(new_res)
            resolutions.add(new_res)
        resolutions = list(resolutions)
        resolutions.sort()
        self.possible_resolutions = resolutions

    def disconnect(self):
        self.capture.release()
        self.is_connected = False
    
    def connect(self):
        self.capture = cv2.VideoCapture(self.port)
        self.is_connected = True

    # def stop_rolling(self):
    #     """
    #     Use .is_rolling as a trigger to kill threads
    #     that are reading in camera data
    #     """
    #     self.stop_rolling_confirmed = False
    #     self.is_rolling = False
    #     while not self.stop_rolling_confirmed:
    #         time.sleep(.1)

# Here I include some helper functions to exhibit/test the functionality 
# of the module
def display_worker(camera, kill_q, win_name=None): 
    if not win_name:
        win_name = f"'q' to quit video {camera.port}"
    
    while True:
        success, frame = camera.capture.read()
        cv2.imshow(win_name, frame)

        if not kill_q.empty():
            _ = kill_q.get()
            cv2.destroyWindow(win_name)
            camera.is_rolling = False
            break

        if cv2.waitKey(1) ==ord('q'):
            cv2.destroyWindow(win_name)
            camera.is_rolling = False
            break
                    
def display(camera, kill_q, win_name=None):
    """ Note, this is just for general observation purposes, 
    while in the process of verifying this module.
    """
    camera.is_rolling = True
    display_thread = Thread(target=display_worker, args= (camera, kill_q, win_name), daemon=True)
    display_thread.start()

# if __name__ == "__main__":
#%%
# if True:

cam1 = Camera(1)
print(cam1.possible_resolutions)
#%%

# cam1.resolution = (752.0, 416.0)
# display(cam1)
# #%%

# cam1.resolution = (1280, 720)
# display(cam1)

# #%%

# cam1.resolution = (1024, 576)
# display(cam1)
#%%

kill_q = queue.Queue()

for res in cam1.possible_resolutions:
    print(f"Testing Resolution {res}")
    cam1.disconnect()
    cam1.connect()
    cam1.resolution = res
    display(cam1, kill_q)
    
    time.sleep(3)
    kill_q.put("End")
    time.sleep(1)
    # cam1.stop_rolling()
    # time.sleep(3)

    # display(cam1)
    
cam1.disconnect()

# %%
