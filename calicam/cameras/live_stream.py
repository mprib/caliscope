# This widget is the primary functional unit of the motion capture. It
# establishes the connection with the video source and manages the thread
# that reads in frames.
import logging

LOG_FILE = "log\live_stream.log"
LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from time import perf_counter, sleep
from queue import Queue
from threading import Thread, Event

import cv2
import numpy as np

from calicam.cameras.camera import Camera


class LiveStream:
    def __init__(self, camera, fps=6):
        self.camera = camera
        self.port = camera.port

        self.reel = Queue(-1)  # infinite size....hopefully doesn't blow up
        self.stop_confirm = Queue()
        self.stop_event = Event()

        self.push_to_reel = False
        self.show_fps = False
        self.set_fps(fps)
        self.FPS_actual = 0
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.roll_camera, args=(), daemon=True)
        self.thread.start()

        # initialize time trackers for actual FPS determination
        self.frame_time = perf_counter()
        self.avg_delta_time = (
            1  # trying to avoid div 0 error...not sure about this though
        )

    def set_fps(self, fps):
        self.fps = fps
        milestones = []
        for i in range(0, fps):
            milestones.append(i / fps)
        logging.info(f"Setting fps to {self.fps}")
        self.milestones = np.array(milestones)

    def wait_to_next_frame(self):
        """based on the target fps, return the time needed to sleep so that
        a frame read immediately after would occur when needed"""

        time = perf_counter()
        fractional_time = time % 1
        all_wait_times = self.milestones - fractional_time
        future_wait_times = all_wait_times[all_wait_times > 0]

        if len(future_wait_times) == 0:
            return 1 - fractional_time
        else:
            return future_wait_times[0]

    def get_FPS_actual(self):
        """set the actual frame rate; called within roll_camera()
        needs to be called from within roll_camera to actually work
        Note that this is a smoothed running average"""
        self.delta_time = perf_counter() - self.start_time
        self.start_time = perf_counter()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time

        # folding in current frame rate to trailing average to smooth out
        self.avg_delta_time = 0.5 * self.avg_delta_time + 0.5 * self.delta_time
        self.previous_time = self.start_time
        return 1 / self.avg_delta_time

    def stop(self):
        self.push_to_reel = False
        self.stop_event.set()
        logging.info(f"Stop signal sent at stream {self.port}")

    def roll_camera(self):
        """
        Worker function that is spun up by Thread. Reads in a working frame,
        calls various frame processing methods on it, and updates the exposed
        frame
        """
        self.start_time = perf_counter() # used to get initial delta_t for FPS
        first_time = True
        while not self.stop_event.is_set():
            if first_time:
                logging.info(f"Camera now rolling at port {self.port}")
                first_time = False

            if self.camera.capture.isOpened():

                # Wait an appropriate amount of time to hit the frame rate target
                sleep(self.wait_to_next_frame())
                read_start = perf_counter()
                self.success, self._working_frame = self.camera.capture.read()

                read_stop = perf_counter()
                self.frame_time = (read_start + read_stop) / 2

                if self.show_fps:
                    self._add_fps()

                if self.push_to_reel and self.success:
                    logging.debug(f"Pushing frame to reel at port {self.port}")
                    self.reel.put([self.frame_time, self._working_frame])

                # Rate of calling recalc must be frequency of this loop
                self.FPS_actual = self.get_FPS_actual()

        logging.info(f"Stream stopped at port {self.port}")
        self.stop_event.clear()
        self.stop_confirm.put("Successful Stop")

    def change_resolution(self, res):

        logging.info(f"About to stop camera at port {self.port}")
        self.stop_event.set()
        self.stop_confirm.get()
        logging.info(f"Roll camera stop confirmed at port {self.port}")

        self.FPS_actual = 0
        self.avg_delta_time = None

        # reconnecting a few times without disconnnect sometimes crashed python
        logging.info(f"Disconnecting from port {self.port}")
        self.camera.disconnect()
        logging.info(f"Reconnecting to port {self.port}")
        self.camera.connect()

        self.camera.resolution = res
        # Spin up the thread again now that resolution is changed
        logging.info(
            f"Beginning roll_camera thread at port {self.port} with resolution {res}"
        )
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
    ports = [0, 1, 2]

    cams = []
    for port in ports:
        print(f"Creating camera {port}")
        cams.append(Camera(port))

    streams = []
    for cam in cams:
        print(f"Creating Video Stream for camera {cam.port}")
        stream = LiveStream(cam)
        stream.push_to_reel = True
        stream.show_fps = True
        streams.append(stream)

    while True:
        try:
            for stream in streams:
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
                print(f"Attempting to change resolution at port {stream.port}")
                stream.change_resolution((1024, 576))

        if key == ord("s"):
            for stream in streams:
                stream.stop()
            cv2.destroyAllWindows()
            exit(0)
