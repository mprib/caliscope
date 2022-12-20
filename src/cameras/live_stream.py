# This widget is the primary functional unit of the motion capture. It
# establishes the connection with the video source and manages the thread
# that reads in frames.
import logging
LOG_FILE = "log\live_stream.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)
import sys
import time as time_module # peculier bug popped up during module testing...perhaps related to conda environment?
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread, Event

import cv2
import mediapipe as mp
import numpy as np

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera


class LiveStream:
    def __init__(self, camera):
        self.camera = camera
        self.port = camera.port

        self.reel = Queue(-1)  # infinite size....hopefully doesn't blow up
        self.shutter_sync = Queue()
        self.stop_event = Event() 

        self.push_to_reel = False
        self.keep_going = True
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.roll_camera, args=(), daemon=True)
        self.thread.start()

        # initialize time trackers for actual FPS determination
        self.frame_time = time_module.perf_counter()
        self.avg_delta_time = 1 # trying to avoid div 0 error...not sure about this though
        

    def get_FPS_actual(self):
        """set the actual frame rate; called within roll_camera()"""
        self.delta_time = time_module.time() - self.start_time
        self.start_time = time_module.time()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time

        # folding in current frame rate to trailing average to smooth out
        self.avg_delta_time = 0.9 * self.avg_delta_time + 0.1 * self.delta_time
        self.previous_time = self.start_time
        self.keep_going = True
        return 1 / self.avg_delta_time
        # TODO: #23 avg_delta_time was zero when testing on the laptop...is this necessary?
    
    def stop(self):
        # self.camera.stop_rolling()
        self.push_to_reel=False
        self.stop_event.set()
        logging.info(f"Stop signal sent at stream {self.port}")
        # self.thread.join()    

    def roll_camera(self):
        """
        Worker function that is spun up by Thread. Reads in a working frame,
        calls various frame processing methods on it, and updates the exposed
        frame
        """
        self.start_time = time_module.time()  # used to get initial delta_t for FPS
        while not self.stop_event.is_set():
            if not self.camera.is_rolling:
                logging.info(f"Camera now rolling at port {self.port}")
            self.camera.is_rolling = True

            if self.camera.capture.isOpened():

                # wait for sync_shutter to fire
                if self.push_to_reel:
                    _ = self.shutter_sync.get()
                    logging.debug(f"Shutter fire signal retrieved at port {self.port}")

                # read in working frame
                read_start = time_module.perf_counter()
                self.status, self._working_frame = self.camera.capture.read()
                read_stop = time_module.perf_counter()
                self.frame_time = (read_start + read_stop) / 2

                if self.push_to_reel:
                    logging.debug(f"Pushing frame to reel at port {self.port}")
                    self.reel.put([self.frame_time, self._working_frame])

                # this may no longer be necessary...consider removing in the future
                # self.frame = self._working_frame.copy()

                # Rate of calling recalc must be frequency of this loop
                self.FPS_actual = self.get_FPS_actual()

                # Stop thread if camera pulls trigger
                if self.camera.stop_rolling_trigger:
                    self.camera.is_rolling = False
                    break
        logging.info(f"Stream stopped at port {self.port}") 

    def change_resolution(self, res):

        # pull cam.stop_rolling_trigger and wait for roll_camera to stop
        logging.info(f"About to stop camera at port {self.port}")
        self.camera.stop_rolling()

        # if the display isn't up and running this may error out (as when trying
        # to initialize the resolution to a non-default value)
        blank_image = np.zeros(self._working_frame.shape, dtype=np.uint8)
        # multiple blank images to account for sync issues
        self.reel.put([time_module.perf_counter(), blank_image])
        self.reel.put([time_module.perf_counter(), blank_image])

        self.FPS_actual = 0
        self.avg_delta_time = None

        # reconnecting a few times without disconnnect sometimes crashed python
        logging.info(f"Disconnecting from port {self.port}")
        self.camera.disconnect()
        logging.info(f"Reconnecting to port {self.port}")
        self.camera.connect()

        self.camera.resolution = res
        # Spin up the thread again now that resolution is changed
        logging.info(f"Beginning roll_camera thread at port {self.port} with resolution {res}")
        self.thread = Thread(target=self.roll_camera, args=(), daemon=True)
        self.thread.start()

    def _add_fps(self):
        """NOTE: this is used in code at bottom, not in external use"""
        self.fps_text = str(int(round(self.FPS_actual, 0)))
        cv2.putText(
            self._working_frame,
            "FPS:" + self.fps_text,
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            3,
        )


if __name__ == "__main__":
    ports = [0]

    cams = []
    for port in ports:
        print(f"Creating camera {port}")
        cams.append(Camera(port))

    streams = []
    for cam in cams:
        print(f"Creating Video Stream for camera {cam.port}")
        stream = LiveStream(cam)
        stream.push_to_reel = True
        stream.shutter_sync.put("Fire")
        stream.shutter_sync.put("Fire")
        # stream.assign_charuco(charuco)
        streams.append(stream)
        
    while True:
        try:
            for stream in streams:
                print(stream.port)
                stream._add_fps()
                stream.shutter_sync.put("Fire")
                time, img = stream.reel.get()
                cv2.imshow(
                    (str(stream.port) + ": 'q' to quit and attempt calibration"),
                    img,
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

        if key == ord("s"):
            for stream in streams:
                stream.stop()
            cv2.destroyAllWindows()
            exit(0)
                
    