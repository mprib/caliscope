# This widget is the primary functional unit of the motion capture. It
# establishes the connection with the video source and manages the thread
# that reads in frames.

import calicam.logger

logger = calicam.logger.get(__name__)

from time import perf_counter, sleep
from queue import Queue
from threading import Thread, Event

import cv2
import numpy as np

from calicam.cameras.camera import Camera
from calicam.cameras.data_packets import FramePacket
from calicam.calibration.charuco import Charuco
from calicam.calibration.corner_tracker import CornerTracker
import calicam.calibration.draw_charuco as draw_charuco


class LiveStream:
    def __init__(self, camera, fps_target=6, charuco=None):
        self.camera = camera
        self.port = camera.port
        self.track_points = Event()

        if charuco is not None:
            self.charuco = charuco
            self.tracker = CornerTracker(charuco)
            self.track_points.set()  # default to tracking points if the charuco is provided
        else:
            self.track_points.clear()  # just to be clear

        self.out_q = Queue(-1)  # infinite size....hopefully doesn't blow up
        self.push_to_out_q = Event()
        self.push_to_out_q.set()  # default behavior is to push to queue
        self.stop_event = Event()

        # make sure camera no longer reading before trying to change resolution
        self.stop_confirm = Queue()

        self._show_fps = False  # used for testing
        self._show_charuco = False  # used for testing

        self.set_fps_target(fps_target)
        self.FPS_actual = 0
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.worker, args=(), daemon=True)
        self.thread.start()

        # initialize time trackers for actual FPS determination
        self.frame_time = perf_counter()
        self.avg_delta_time = 1  # initialize to something to avoid errors elsewhere

    def set_fps_target(self, fps):
        """
        This is done through a method as it will also do a one-time determination of the times as which
        frames should be read (the milestones)
        """

        self.fps = fps
        milestones = []
        for i in range(0, fps):
            milestones.append(i / fps)
        logger.info(f"Setting fps to {self.fps}")
        self.milestones = np.array(milestones)

    def update_charuco(self, charuco: Charuco):
        self.charuco = charuco
        self.tracker = CornerTracker(charuco)

    def wait_to_next_frame(self):
        """
        based on the next milestone time, return the time needed to sleep so that
        a frame read immediately after would occur when needed
        """

        time = perf_counter()
        fractional_time = time % 1
        all_wait_times = self.milestones - fractional_time
        future_wait_times = all_wait_times[all_wait_times > 0]

        if len(future_wait_times) == 0:
            return 1 - fractional_time
        else:
            return future_wait_times[0]

    def get_FPS_actual(self):
        """
        set the actual frame rate; called within roll_camera()
        needs to be called from within roll_camera to actually work
        Note that this is a smoothed running average
        """
        self.delta_time = perf_counter() - self.start_time
        self.start_time = perf_counter()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time

        # folding in current frame rate to trailing average to smooth out
        self.avg_delta_time = 0.5 * self.avg_delta_time + 0.5 * self.delta_time
        self.previous_time = self.start_time
        return 1 / self.avg_delta_time

    def stop(self):
        self.push_to_out_q.clear()
        self.stop_event.set()
        logger.info(f"Stop signal sent at stream {self.port}")

    def worker(self):
        """
        Worker function that is spun up by Thread. Reads in a working frame,
        calls various frame processing methods on it, and updates the exposed
        frame
        """
        self.start_time = perf_counter()  # used to get initial delta_t for FPS
        first_time = True
        while not self.stop_event.is_set():
            if first_time:
                logger.info(f"Camera now rolling at port {self.port}")
                first_time = False

            if self.camera.capture.isOpened():

                # slow wait if not pushing frames                
                # this is a sub-optimal busy wait spin lock, but it works and I'm tired.
                while not self.push_to_out_q.is_set():
                    sleep(.2)

                # Wait an appropriate amount of time to hit the frame rate target
                sleep(self.wait_to_next_frame())

                read_start = perf_counter()
                self.success, self.frame = self.camera.capture.read()

                read_stop = perf_counter()
                point_data = None # Provide initial value here...may get overwritten
                self.frame_time = (read_start + read_stop) / 2

                if self.push_to_out_q.is_set() and self.success:
                    logger.debug(f"Pushing frame to reel at port {self.port}")

                    if self.track_points.is_set():
                        point_data = self.tracker.get_points(self.frame)
                    # else:
                        # point_data = None

                if self._show_fps:
                    self._add_fps()

                frame_packet = FramePacket(
                    port=self.port,
                    frame_time=self.frame_time,
                    frame=self.frame,
                    points=point_data,
                )

                if self._show_charuco:
                    draw_charuco.corners(frame_packet)
                    # self.out_q.put([self.frame_time, self.frame])

                if self.push_to_out_q.is_set():
                    self.out_q.put(frame_packet)

                # Rate of calling recalc must be frequency of this loop
                self.FPS_actual = self.get_FPS_actual()

        logger.info(f"Stream stopped at port {self.port}")
        self.stop_event.clear()
        self.stop_confirm.put("Successful Stop")

    def change_resolution(self, res):

        logger.info(f"About to stop camera at port {self.port}")
        self.stop_event.set()
        self.stop_confirm.get()
        logger.info(f"Roll camera stop confirmed at port {self.port}")

        self.FPS_actual = 0
        self.avg_delta_time = None

        # reconnecting a few times without disconnnect sometimes crashed python
        logger.info(f"Disconnecting from port {self.port}")
        self.camera.disconnect()
        logger.info(f"Reconnecting to port {self.port}")
        self.camera.connect()

        self.camera.resolution = res
        # Spin up the thread again now that resolution is changed
        logger.info(
            f"Beginning roll_camera thread at port {self.port} with resolution {res}"
        )
        self.thread = Thread(target=self.worker, args=(), daemon=True)
        self.thread.start()

    def _add_fps(self):
        """NOTE: this is used in F5 test, not in external use"""
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


if __name__ == "__main__":
    # ports = [2, 3, 4]
    ports = [3]

    cams = []
    for port in ports:
        print(f"Creating camera {port}")
        cam = Camera(port)
        cam.exposure = -7
        cams.append(cam)

    # standard inverted charuco
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    streams = []
    for cam in cams:
        print(f"Creating Video Stream for camera {cam.port}")
        stream = LiveStream(cam, fps_target=12, charuco=charuco)
        stream.push_to_out_q.set()
        stream._show_fps = True
        stream._show_charuco = True
        streams.append(stream)

    while True:
        try:
            for stream in streams:
                frame_packet = stream.out_q.get()

                cv2.imshow(
                    (str(frame_packet.port) + ": 'q' to quit and attempt calibration"),
                    frame_packet.frame,
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
