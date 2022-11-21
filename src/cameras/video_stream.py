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
    def __init__(self, cam):
        self.cam = cam
        self.reel = Queue(-1)  # infinite size....hopefully doesn't blow up
        self.push_to_reel = False

        # Start the thread to read frames from the video stream
        self.cap_thread = Thread(target=self.roll_camera, args=(), daemon=True)
        self.cap_thread.start()
        self.frame_name = "Cam" + str(cam.port)

        # initialize time trackers for actual FPS determination
        self.frame_time = time.perf_counter()
        self.avg_delta_time = None

    def set_shutter_sync(self, shutter_sync):
        """shutter sync is a thread queue that triggers end of wait cycle"""
        self.shutter_sync = shutter_sync

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

    def apply_rotation(self):

        if self.cam.rotation_count == 0:
            pass
        elif self.cam.rotation_count in [1, -3]:
            self._working_frame = cv2.rotate(
                self._working_frame, cv2.ROTATE_90_CLOCKWISE
            )
        elif self.cam.rotation_count in [2, -2]:
            self._working_frame = cv2.rotate(self._working_frame, cv2.ROTATE_180)
        elif self.cam.rotation_count in [-1, 3]:
            self._working_frame = cv2.rotate(
                self._working_frame, cv2.ROTATE_90_COUNTERCLOCKWISE
            )

    def roll_camera(self):
        """
        Worker function that is spun up by Thread. Reads in a working frame,
        calls various frame processing methods on it, and updates the exposed
        frame
        """
        self.start_time = time.time()  # used to get initial delta_t for FPS
        while True:
            self.cam.is_rolling = True

            # note this line is truly necessary otherwise error upon closing capture
            if self.cam.capture.isOpened():

                # wait for sync_shutter to fire
                if self.push_to_reel:
                    _ = self.shutter_sync.get()

                # read in working frame
                read_start = time.perf_counter()
                self.status, self._working_frame = self.cam.capture.read()
                read_stop = time.perf_counter()
                self.frame_time = (read_start + read_stop) / 2

                # REAL TIME OVERLAYS ON self._working_frame
                # self.run_mediapipe_hands()
                # self.process_charuco()

                # I have misgivings about including this in here
                # should be used as a sanity check of distortion params
                # applied sparingly and never run when doing *anything* else
                # self.apply_undistortion()

                # must apply rotation at end...
                # otherwise mismatch in frame / grid history dimensions
                self.apply_rotation()

                if self.push_to_reel:
                    # print(f"Pushing from port {self.cam.port} at {self.frame_time}")
                    self.reel.put(
                        [
                            self.frame_time,
                            self._working_frame,
                            # self.mono_cal._frame_corner_ids,
                            # self.mono_cal._frame_corners,
                            # self.mono_cal.board_FOR_corners,
                        ]
                    )

                # update frame that is emitted to GUI by frame emitter
                # note: frame_emitter uses a throttled loop to just periodically
                # read the current frame. It's not trying to by precise or
                # pick up every frame
                self.frame = self._working_frame.copy()

                # Rate of calling recalc must be frequency of this loop
                self.FPS_actual = self.get_FPS_actual()

                # Stop thread if camera pulls trigger
                if self.cam.stop_rolling_trigger:
                    self.cam.is_rolling = False
                    break

    def change_resolution(self, res):
        # pull cam.stop_rolling_trigger and wait for roll_camera to stop
        self.cam.stop_rolling()

        # if the display isn't up and running this may error out (as when trying
        # to initialize the resolution to a non-default value)
        try:
            blank_image = np.zeros(self.frame.shape, dtype=np.uint8)
            self.frame = blank_image
        except:
            pass

        self.FPS_actual = 0
        self.avg_delta_time = None

        # reconnecting a few times without disconnnect sometimes crashed python
        self.cam.disconnect()
        self.cam.connect()

        self.cam.resolution = res
        # if self.mono_calib:
        # try:
        # self.mono_cal.initialize_grid_history()
        # except:
        # pass

        # Spin up the thread again now that resolution is changed
        self.cap_thread = Thread(target=self.roll_camera, args=(), daemon=True)
        self.cap_thread.start()

    # def toggle_mediapipe(self):
    # self.show_mediapipe = not self.show_mediapipe

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

    # def assign_charuco(self, charuco):
    # self.mono_cal = MonoCalibrator(self.cam, charuco)

    # def process_charuco(self):
    # """Heavy lifting from the charuco module. This method could involve just
    # displaying the identified corners on the frame, or adding them to
    # the list of corners for running a calibration.

    # The scope of the action depends on setting flags for:
    # self.track_charuco
    # self.collect_charuco_corners
    # """
    #     if self.track_charuco:
    #         self.mono_cal.track_corners(self._working_frame, self.frame_time)
    #         if self.collect_charuco_corners:
    #             self.mono_cal.collect_corners()

    #         self._working_frame = self.mono_cal.merged_grid_history()

    # def apply_undistortion(self):

    #     if self.undistort == True:  # and self.mono_cal.is_calibrated:
    #         self._working_frame = cv2.undistort(
    #             self._working_frame, self.cam.camera_matrix, self.cam.distortion
    #         )


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

        # toggle mediapipe with 'm'
        # if key == ord("m"):
        #     print("Toggling Mediapipe")
        #     for stream in streams:
        #         print(stream.frame_name)
        #         stream.toggle_mediapipe()

        if key == ord("r"):
            print("Rotate Frame CW")

            for stream in streams:
                stream.cam.rotate_CW()
                print(stream.frame_name + " " + str(stream.cam.rotation_count))

        if key == ord("l"):
            print("Rotate Frame CCW")

            for stream in streams:
                stream.cam.rotate_CCW()
                print(stream.frame_name + " " + str(stream.cam.rotation_count))

        # Toggle charuco display
        # if key == ord("c"):
        #     for stream in streams:
        #         stream.track_charuco = not stream.track_charuco

        # # Toggle charuco display
        # if key == ord("C"):
        #     for stream in streams:
        #         stream.charuco_being_traced = True
        #         stream.collect_charuco_corners = not stream.collect_charuco_corners

        # # Toggle undistortion
        # if key == ord("d"):
        #     for stream in streams:
        #         stream.undistort = not stream.undistort

        # 'q' to quit
        if key == ord("q"):
            for stream in streams:
                # try:
                #     stream.mono_cal.calibrate()
                # except:
                #     pass
                stream.cam.capture.release()
            cv2.destroyAllWindows()
            exit(0)

        if key == ord("v"):
            for stream in streams:
                stream.change_resolution((1280, 720))
