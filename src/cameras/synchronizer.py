# This is a very much work in progress to sort through coding up the basic
# rule for frame syncing that I have in mind. Put all the frames in the first
# index. If one frame was read after the earliest frame in the next index,
# move it to the next index

import logging

LOG_FILE = "synchronizer.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# from src.cameras.video_stream import VideoStream


class Synchronizer:
    streams: list

    def __init__(self, streams: dict, fps_target):
        self.streams = streams

        self.ports = []

        # place to pull real time data with get()
        # self.synced_frames_q = Queue()
        self.current_bundle = None

        self.subscriptions = []
        # initialize frame data which will hold everything pushed from
        # roll_camera() for each port
        self.frame_data = {}
        for port, stream in self.streams.items():
            self.ports.append(port)
            # self.frame_data[port] = []

        self.port_frame_count = {port: 0 for port in self.ports}
        self.port_current_frame = {port: 0 for port in self.ports}

        self.frame_rates = []
        self.mean_frame_times = []

        self.shutter_sync = Queue()
        self.fps_target = fps_target
        self.throttle_wait = 1 / fps_target  # initial value that will get revised

        logging.info("About to submit Threadpool of frame Harvesters")
        self.threads = []
        for port, stream in self.streams.items():
            stream.set_shutter_sync(self.shutter_sync)

            t = Thread(target=self.harvest_frames, args=(stream,), daemon=True)
            t.start()
            self.threads.append(t)
        logging.info("Frame harvesters just submitted")

        logging.info("Starting frame bundler...")
        self.bundler = Thread(target=self.bundle_frames, args=(), daemon=True)
        self.bundler.start()

    def subscribe(self, q):
        self.subscriptions.append(q)

    def harvest_frames(self, device):
        # pull data from the
        port = device.cam.port
        device.push_to_reel = True
        device.track_charuco = True

        logging.info(f"Beginning to collect data generated at port {port}")
        frame_index = 0

        while True:

            (
                frame_time,
                frame,
            ) = device.reel.get()

            self.frame_data[f"{port}_{frame_index}"] = {
                "port": port,
                "frame": frame,
                "frame_index": frame_index,
                "frame_time": frame_time,
            }

            frame_index += 1
            self.port_frame_count[port] = frame_index

    # get minimum value of frame_time for next layer
    def earliest_next_frame(self):
        """Looks at next unassigned frame across the ports to determine
        the earliest time at which each of them was read"""
        time_of_next_frames = []
        for port in self.ports:
            next_index = self.port_current_frame[port] + 1
            next_frame_time = self.frame_data[f"{port}_{next_index}"]["frame_time"]
            time_of_next_frames.append(next_frame_time)

        return min(time_of_next_frames)

    def frame_slack(self):
        """Determine how many unassigned frames are sitting in self.dataframe"""

        slack = [
            self.port_frame_count[port] - self.port_current_frame[port]
            for port in self.ports
        ]
        logging.debug(f"Slack in frames is {slack}")
        return min(slack)

    def average_fps(self):
        """"""
        # only look at the most recent layers
        if len(self.mean_frame_times) > 10:
            self.mean_frame_times = self.mean_frame_times[-10:]

        delta_t = np.diff(self.mean_frame_times)
        mean_delta_t = np.mean(delta_t)

        return 1 / mean_delta_t

    def throttle_fps(self):
        fps = self.average_fps()
        # if fps > self.fps_target:
        if abs(fps - self.fps_target) < 1:
            self.throttle_wait += 0.1 * ((1 / self.fps_target) - (1 / fps))
        else:
            self.throttle_wait += 0.5 * ((1 / self.fps_target) - (1 / fps))
        if np.isnan(fps):
            self.throttle_wait = 0
        print(f"FPS: {fps}")
        # print(self.throttle_wait)
        time.sleep(max(self.throttle_wait, 0))

    def bundle_frames(self):

        logging.info(f"Waiting for all ports to begin harvesting corners...")
        while self.frame_slack() == 0:
            self.shutter_sync.put("fire")
            time.sleep(0.01)

        # need to have 2 frames to assess bundling
        for port in self.ports:
            self.shutter_sync.put("fire")

        logging.info("About to start bundling frames...")
        while True:
            # Trigger device to proceed with reading frame and pushing to reel
            for port in self.ports:
                self.shutter_sync.put("fire")

            # wait for frame data to populate
            while self.frame_slack() < 2:
                # self.shutter_sync.put("fire")
                time.sleep(0.01)

            # don't put a frame in a bundle if the next bundle has a frame before it
            # bundle_start = self.earliest_current_frame()
            bundle_cutoff_time = self.earliest_next_frame()

            # delta_t = bundle_cutoff_time - bundle_start
            # self.frame_rates.append(1/delta_t)

            # only throttle if you are mostly current to avoid progressive
            # drift in lag
            if self.frame_slack() < 3:
                self.throttle_fps()

            next_layer = {}
            layer_frame_times = []
            for port in self.ports:
                current_frame_index = self.port_current_frame[port]

                port_index_key = f"{port}_{current_frame_index}"
                current_frame_data = self.frame_data[port_index_key]
                frame_time = current_frame_data["frame_time"]

                if frame_time < bundle_cutoff_time:
                    # add the data and increment the index
                    next_layer[port] = self.frame_data.pop(port_index_key)
                    self.port_current_frame[port] += 1
                    layer_frame_times.append(frame_time)
                else:
                    next_layer[port] = None
            logging.debug(f"Unassigned Frames: {len(self.frame_data)}")

            # commenting out previous framework code...no longer passing frames
            # via queue, just passing notification that frame is updated
            self.mean_frame_times.append(np.mean(layer_frame_times))
            # self.synced_frames_q.put(next_layer)

            self.current_bundle = next_layer
            for q in self.subscriptions:
                q.put("new bundle available")


if __name__ == "__main__":

    # DON"T DEAL WITH THE SESSION OBJECT IN TESTS...ONLY MORE FOUNDATIONAL ELEMENTS
    from src.cameras.camera import Camera
    from src.cameras.video_stream import VideoStream
    from src.session import Session

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "default_session")

    cameras = []
    ports = [0, 1, 2]
    for port in ports:
        cameras.append(Camera(port))

    streams = {}
    for cam in cameras:
        streams[cam.port] = VideoStream(cam)

    syncr = Synchronizer(streams, fps_target=30)

    notification_q = Queue()

    syncr.subscriptions.append(notification_q)

    all_bundles = []
    while True:
        frame_bundle_notice = notification_q.get()
        print(frame_bundle_notice)
        for port, frame_data in syncr.current_bundle.items():
            if frame_data:
                cv2.imshow(f"Port {port}", frame_data["frame"])

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
