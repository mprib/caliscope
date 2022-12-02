# This widget is the primary functional unit of the motion capture. It
# establishes the connection with the video source and manages the thread
# that reads in frames.

import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera


class VideoStream:
    def __init__(self, camera):
        self.camera = camera
        self.reel = Queue(-1)  # infinite size....hopefully doesn't blow up
        self.push_to_reel = False

        # Start the thread to read frames from the video stream
        self.cap_thread = Thread(target=self.roll_camera, args=(), daemon=True)
        self.cap_thread.start()
        self.frame_name = "Cam" + str(camera.port)

        # initialize time trackers for actual FPS determination
        self.frame_time = time.perf_counter()
        self.avg_delta_time = 1
        
        self.shutter_sync = Queue()

    def get_FPS_actual(self):
        """set the actual frame rate; called within roll_camera()"""
        self.delta_time = time.time() - self.start_time
        self.start_time = time.time()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time

        # folding in current frame rate to trailing average to smooth out
        self.avg_delta_time = 0.9 * self.avg_delta_time + 0.1 * self.delta_time
        self.previous_time = self.start_time

        return 1 / self.avg_delta_time
        # TODO: #23 avg_delta_time was zero when testing on the laptop...is this necessary?
    def roll_camera(self):
        """
        Worker function that is spun up by Thread. Reads in a working frame,
        calls various frame processing methods on it, and updates the exposed
        frame
        """
        self.start_time = time.time()  # used to get initial delta_t for FPS
        while True:
            self.camera.is_rolling = True
            if self.camera.capture.isOpened():

                # wait for sync_shutter to fire
                if self.push_to_reel:
                    _ = self.shutter_sync.get()

                # read in working frame
                read_start = time.perf_counter()
                self.status, self._working_frame = self.camera.capture.read()
                read_stop = time.perf_counter()
                self.frame_time = (read_start + read_stop) / 2

                if self.push_to_reel:
                    self.reel.put([self.frame_time, self._working_frame])

                # this may no longer be necessary...consider removing in the future
                # self.frame = self._working_frame.copy()

                # Rate of calling recalc must be frequency of this loop
                self.FPS_actual = self.get_FPS_actual()

                # Stop thread if camera pulls trigger
                if self.camera.stop_rolling_trigger:
                    self.camera.is_rolling = False
                    break

    def change_resolution(self, res):
        # pull cam.stop_rolling_trigger and wait for roll_camera to stop
        self.camera.stop_rolling()

        # if the display isn't up and running this may error out (as when trying
        # to initialize the resolution to a non-default value)
        blank_image = np.zeros(self._working_frame.shape, dtype=np.uint8)
        # multiple blank images to account for sync issues
        self.reel.put([time.perf_counter(), blank_image])
        self.reel.put([time.perf_counter(), blank_image])

        self.FPS_actual = 0
        self.avg_delta_time = None

        # reconnecting a few times without disconnnect sometimes crashed python
        self.camera.disconnect()
        self.camera.connect()

        self.camera.resolution = res
        # Spin up the thread again now that resolution is changed
        self.cap_thread = Thread(target=self.roll_camera, args=(), daemon=True)
        self.cap_thread.start()

    def _add_fps(self):
        """NOTE: this is used in code at bottom, not in external use"""
        self.fps_text = str(int(round(self.FPS_actual, 0)))
        cv2.putText(
            self.frame,
            "FPS:" + self.fps_text,
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            3,
        )


# Highlight module functionality. View a frame with mediapipe hands
# press "q" to quit
if __name__ == "__main__":
    ports = [0]

    cams = []
    for port in ports:
        print(f"Creating camera {port}")
        cams.append(Camera(port))

    streams = []
    for cam in cams:
        print(f"Creating Video Stream for camera {cam.port}")
        stream = VideoStream(cam)
        # stream.assign_charuco(charuco)
        streams.append(stream)

    while True:
        try:
            for stream in streams:
                stream._add_fps()
                cv2.imshow(
                    str(stream.frame_name + ": 'q' to quit and attempt calibration"),
                    stream.frame,
                )

        # bad reads until connection to src established
        except AttributeError:
            pass

        key = cv2.waitKey(1)

        if key == ord("q"):
            for stream in streams:
                stream.camera.capture.release()
            cv2.destroyAllWindows()
            exit(0)

        if key == ord("v"):
            for stream in streams:
                stream.change_resolution((1280, 720))
