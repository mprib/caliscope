# The purpose of this module is to create a class that will operate as an
# interface for a given camera. It will  store data related to
# a variety of camera specific variables and allow setting of these variables:
#
# - port
#   - an integer for a live camera
#   - note this is focused on live feeds...pre-recorded video doesn't need this
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

import platform
import time
import os

import cv2
import caliscope.logger
logger = caliscope.logger.get(__name__)

TEST_FRAME_COUNT = 10
MIN_RESOLUTION_CHECK = 200
MAX_RESOLUTION_CHECK = 10000


class Camera:

    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # see above for constants used to access properties
    def __init__(self, port, verified_resolutions = None, connect_API = None):

        if connect_API is not None:
            self.connect_API = connect_API
        else:
            if os.name == 'nt': #windows
                self.connect_API = cv2.CAP_DSHOW
            else: # UNIX variant
                self.connect_API = cv2.CAP_ANY

        # check if source has a data feed before proceeding...if not it is
        # either in use or fake
        logger.info(f"Attempting to connect video capture at port {port}")
        test_capture = cv2.VideoCapture(port,self.connect_API)
        for _ in range(0, TEST_FRAME_COUNT):
            good_read, frame = test_capture.read()

            # pass # dealing with this in the else statemetn below...not a real camera
        if good_read:
            logger.info(f"Good read at port {port}...proceeding")
            self.port = port
            self.capture = test_capture
            self.active_port = True
            # limit buffer size so that you are always reading the latest frame
            self.capture.set(
                cv2.CAP_PROP_BUFFERSIZE, 1
            )  # from https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-getanother thread signaled a change to mediapipe overley-sync

            self.ignore = False # flag camera during single camera setup to be ignored in the future

            # sets orientation in the GUI, but otherwise does not affect the frame
            self.rotation_count = 0  # +1 for each 90 degree CW rotation, -1 for CCW

            self.set_exposure()
            self.set_default_resolution()
        else: 
            # probably busy
            logger.info(f"Camera at port {port} appears to be busy")
            self.port = port
            self.capture = None
            self.active_port = False
            raise Exception(f"Not reading at port {port}...likely in use")

        # Test to see if camera is virtual
        spoof_resolution = (599,599)
        self.size = spoof_resolution
        if self.size == spoof_resolution:
            self.virtual_camera = True
        else:
            self.virtual_camera = False

        if not self.virtual_camera:
            if verified_resolutions is None:
                self.set_possible_resolutions()
            else:
                self.verified_resolutions = verified_resolutions
            # camera initializes as uncalibrated
            self.error = None
            self.matrix = None
            self.distortions = None
            self.grid_count = None
            self.translation = None
            self.rotation = None
        if isinstance(self.verified_resolutions[0], int):
            # probably not real
            self.port = port
            self.capture = None
            self.active_port = False
            logger.info(f"Camera at port {port} may be virtual")
            raise Exception(f"{port}...likely not real")

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        """Note that OpenCV appears to change the exposure value, but
        this is not read back accurately through the getter, so just
        track it manually after updating"""
        if platform.system()=="Windows":
            self.capture.set(cv2.CAP_PROP_EXPOSURE, value)
        
        else:
            self.capture.set(cv2.CAP_PROP_IOS_DEVICE_EXPOSURE, value)

        self._exposure = value

    @property
    def _width(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @_width.setter
    def _width(self, value):
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, value)

    @property
    def _height(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @_height.setter
    def _height(self, value):
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, value)

    @property
    def size(self):
        return (self._width, self._height)

    @size.setter
    def size(self, value):
        """Currently, this is how the resolution is actually changed"""
        self._width = value[0]
        self._height = value[1]

    def set_default_resolution(self):
        """called at initilization before anything has changed"""
        self.default_resolution = self.size

    def set_exposure(self):
        """Need an initial value, though it does not appear that updates to
        exposure reliably read back"""
        self._exposure = self.capture.get(cv2.CAP_PROP_EXPOSURE)
        self.exposure = self._exposure  # port seemed to hold on to old exposure

    def get_nearest_resolution(self, test_width):
        """This strange little method just temporarly stores the current value
        of the resolution to be replaced at the end, then tries a value
        and then reads what resolution closest to it the capture offers,
        then returns the capture to its original state"""
        old_width = self._width
        self._width = test_width
        resolution = self.size
        self._width = old_width
        return resolution

    def set_possible_resolutions(self):
        self.verified_resolutions = []
        for resolution in RESOLUTIONS_TO_CHECK:
            # attempt to set the camera to the given resolution
            logger.info(f"Checking resolution of {resolution} at port {self.port}")
            self.size = resolution
            
            # if it sticks, then that resolution is verified
            if resolution == self.size:
                self.verified_resolutions.append(resolution)
                

    def rotate_CW(self):
        if self.rotation_count == 3:
            self.rotation_count = 0
        else:
            self.rotation_count = self.rotation_count + 1

    def rotate_CCW(self):
        if self.rotation_count == -3:
            self.rotation_count = 0
        else:
            self.rotation_count = self.rotation_count - 1

    def disconnect(self):
        self.capture.release()

    def connect(self):
        self.capture = cv2.VideoCapture(self.port, self.connect_API)

    def calibration_summary(self):
        # Calibration output presented in label on far right
        grid_count = "Grid Count:\t" + str(self.grid_count)
        size_text = "Resolution:\t" + str(self.size[0]) + "x" + str(self.size[1])

        # only grab if they exist
        if self.error and self.error != "NA":
            error_text = f"Error:\t{round(self.error,3)} "
            cam_matrix_text = "Camera Matrix:\n" + (
                "\n".join(
                    [
                        "\t".join([str(round(float(cell), 1)) for cell in row])
                        for row in self.matrix
                    ]
                )
            )
            distortion_text = "Distortion:\t" + ",".join(
                [str(round(float(cell), 2)) for cell in self.distortions]
            )

            # print(self.camera_matrix)
            summary = (
                grid_count
                + "\n\n"
                + error_text
                + "\n\n"
                + size_text
                + "\n\n"
                + cam_matrix_text
                + "\n\n"
                + distortion_text
            )
            return summary
        else:
            return "No Calibration Stored"

# common possibilities taken from https://en.wikipedia.org/wiki/List_of_common_resolutions
RESOLUTIONS_TO_CHECK = [
    # (352, 240),
    # (352, 288),
    # (352, 480),
    # (352, 576),
    # (352, 480),
    # (480, 480),
    # (480, 576),
    # (528, 480),
    # (544, 480),
    # (544, 576),
    (640, 480),
    # (704, 480),
    # (704, 576),
    # (720, 480),
    # (720, 576),
    # (720, 480),
    # (720, 576),
    (1280, 720),
    # (1280, 1080),
    # (1440, 1080),
    (1920, 1080),
    # (3840, 2160),
    # (7680, 4320),
]

######################### TEST FUNCTIONALITY OF CAMERAS ########################
if __name__ == "__main__":

    cam = Camera(4, verified_resolutions=[(640,480), (1280,720)])
    logger.info(f"Camera {cam.port} has possible resolutions: {cam.verified_resolutions}")

    for res in cam.verified_resolutions:
        logger.info(f"Testing Resolution {res}")
        logger.info("Disconnecting from camera")
        cam.disconnect()
        logger.info("Reconnecting to camera")
        cam.connect()
        logger.info(f"Setting camera size to {res}")
        cam.size = res
        logger.info("Resolution successfully updated...")
            
        while True:
            success, frame = cam.capture.read()
            cv2.imshow(f"Resolution: {res}; press 'q' to move to next resolution", frame)
            if cv2.waitKey(1) == ord("q"):
                cam.disconnect()
                cv2.destroyAllWindows()
                break

    cam.connect()

    # while not cam.capture.isOpened():
    #     time.sleep(.01)

    exposure_test_started = False
    
    start_time = time.perf_counter()
    
    while True:
        success, frame = cam.capture.read()
        elapsed_seconds = int(time.perf_counter()-start_time)
        logger.debug(f"{elapsed_seconds} seconds have elapsed since loop began")

        cv2.imshow("Exposure Test", frame)
       
        cam.exposure = -10+elapsed_seconds 
         
        if cv2.waitKey(1) == ord("q"):
            cam.disconnect()
            cv2.destroyAllWindows()
            break     
        
        if elapsed_seconds > 10:
            cam.disconnect()
            cv2.destroyAllWindows()
            break 

# %%
