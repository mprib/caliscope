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

import cv2
from threading import Thread

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
            self.stream_active = False
            self.show_me_active = False

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
        # print("Setting Exposure")
        self.capture.set(cv2.CAP_PROP_EXPOSURE, value)
        self._exposure = value

    @property
    def _width(self):
        # print("Getting _width")
        return self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    @_width.setter
    def _width(self, value):
        # print("Setting _width")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, value)


    @property
    def _height(self):
        return self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @_height.setter
    def _height(self, value):
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, value)

    @property
    def resolution(self):
        # print("Getting Resolution")
        return (self._width, self._height)

    @resolution.setter
    def resolution(self, value):
        if self.stream_active:
            # self.stop_q.put("Stop")
            self.capture.release()
            self.capture = cv2.VideoCapture(self.port)
        
        self._width = value[0]
        self._height = value[1]
        

    def show_me_worker(self, win_name=None): 
        if not win_name:
            win_name = f"'q' to quit video {self.port}"
        
        while True:

            success, frame = self.capture.read()
            cv2.imshow(win_name, frame)

            if cv2.waitKey(1) ==ord('q'):
                cv2.destroyWindow(win_name)
                break
                        

    def show_me(self, win_name=None):
        """ Note, this is just for general observation purposes, 
        I need to see how things work in the display widget where
        it actually matters. This may then neceessitate some other method
        of handling things."""
        stream_thread = Thread(target=self.show_me_worker, args= (win_name, ), daemon=True)
        stream_thread.start()

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
        # reminder on implementation: calling property getter of _width
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
cam1 = Camera(1)

#%% 
cam1.show_me()
