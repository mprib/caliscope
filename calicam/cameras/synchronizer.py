import logging

LOG_FILE = "log\synchronizer.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread, Event

import cv2
import numpy as np


class Synchronizer:
    def __init__(self, streams: dict, fps_target=6):
        self.streams = streams
        self.current_synched_frames = None

        self.synch_notice_subscribers = (
            []
        )  # queues that will be notified of new synched frames
        self.synched_frames_subscribers = (
            []
        )  # queues that will receive actual frame data

        self.frame_data = {}
        self.stop_event = Event()

        self.ports = []
        for port, stream in self.streams.items():
            self.ports.append(port)

        self.fps_target = fps_target
        self.update_fps_targets(fps_target)
        self.fps = fps_target

        self.initialize_ledgers()
        self.spin_up()

    def update_fps_targets(self, target):
        logging.info(f"Attempting to change target fps in streams to {target}")
        for port, stream in self.streams.items():
            stream.set_fps(target)

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        for t in self.threads:
            t.join()

    def initialize_ledgers(self):

        self.port_frame_count = {port: 0 for port in self.ports}
        self.port_current_frame = {port: 0 for port in self.ports}
        self.mean_frame_times = []

    def spin_up(self):

        logging.info("About to submit Threadpool of frame Harvesters")
        self.threads = []
        for port, stream in self.streams.items():
            t = Thread(target=self.harvest_frames, args=(stream,), daemon=True)
            t.start()
            self.threads.append(t)
        logging.info("Frame harvesters just submitted")

        logging.info("Starting frame synchronizer...")
        self.thread = Thread(target=self.synch_frames_worker, args=(), daemon=True)
        self.thread.start()

    def subscribe_to_notice(self, q):
        # subscribers are notified via the queue that new frames are available
        # this is intended to avoid issues with latency due to multiple iterations
        # of frames being passed from one queue to another
        logging.info("Adding queue to receive notice of synched frames update")
        self.synch_notice_subscribers.append(q)

    def subscribe_to_synched_frames(self, q):
        logging.info("Adding queue to receive synched frames")
        self.synched_frames_subscribers.append(q)

    def release_synched_frames_q(self, q):
        logging.info("Releasing record queue")
        self.synched_frames_subscribers.remove(q)

    def harvest_frames(self, stream):
        port = stream.port
        stream.push_to_reel = True

        logging.info(f"Beginning to collect data generated at port {port}")

        while not self.stop_event.is_set():
            frame_index = self.port_frame_count[port]

            (
                frame_time,
                frame,
            ) = stream.reel.get()

            if frame_time == -1:  # signal from recorded stream that end of file reached
                break

            self.frame_data[f"{port}_{frame_index}"] = {
                "port": port,
                "frame": frame,
                "frame_index": frame_index,
                "frame_time": frame_time,
            }

            logging.debug(
                f"Frame data harvested from reel {port} with index {frame_index} and frame time of {frame_time}"
            )
            self.port_frame_count[port] += 1

        logging.info(f"Frame harvester for port {port} completed")

    # get minimum value of frame_time for next layer
    def earliest_next_frame(self, port):
        """Looks at next unassigned frame across the ports to determine
        the earliest time at which each of them was read"""
        times_of_next_frames = []
        for p in self.ports:
            next_index = self.port_current_frame[p] + 1
            frame_data_key = f"{p}_{next_index}"

            # problem with outpacing the threads reading data in, so wait if need be
            while frame_data_key not in self.frame_data.keys():
                logging.debug(
                    f"Waiting in a loop for frame data to populate with key: {frame_data_key}"
                )
                time.sleep(0.001)

            next_frame_time = self.frame_data[frame_data_key]["frame_time"]
            if p != port:
                times_of_next_frames.append(next_frame_time)

        return min(times_of_next_frames)

    def latest_current_frame(self, port):
        """Provides the latest frame_time of the current frames not inclusive of the provided port"""
        times_of_current_frames = []
        for p in self.ports:
            current_index = self.port_current_frame[p]
            current_frame_time = self.frame_data[f"{p}_{current_index}"]["frame_time"]
            if p != port:
                times_of_current_frames.append(current_frame_time)

        return max(times_of_current_frames)

    def frame_slack(self):
        """Determine how many unassigned frames are sitting in self.dataframe"""
        slack = [
            self.port_frame_count[port] - self.port_current_frame[port]
            for port in self.ports
        ]
        logging.debug(f"Slack in frames is {slack}")
        return min(slack)

    def average_fps(self):
        if len(self.mean_frame_times) > 10:  # only looking at the most recent layers
            self.mean_frame_times = self.mean_frame_times[-10:]

        delta_t = np.diff(self.mean_frame_times)
        mean_delta_t = np.mean(delta_t)

        return 1 / mean_delta_t

    def synch_frames_worker(self):

        logging.info(f"Waiting for all ports to begin harvesting corners...")

        sync_time = time.perf_counter()

        sync_index = 0

        logging.info("About to start synchronizing frames...")
        while not self.stop_event.is_set():

            next_layer = {}
            layer_frame_times = []

            # build earliest next/latest current dictionaries for each port to determine where to put frames
            # must be done before going in and making any updates to the frame index
            earliest_next = {}
            latest_current = {}

            for port in self.ports:
                earliest_next[port] = self.earliest_next_frame(port)
                latest_current[port] = self.latest_current_frame(port)
                current_frame_index = self.port_current_frame[port]

            for port in self.ports:
                current_frame_index = self.port_current_frame[port]

                port_index_key = f"{port}_{current_frame_index}"
                current_frame_data = self.frame_data[port_index_key]
                frame_time = current_frame_data["frame_time"]

                # don't put a frame in a synched frame packet if the next packet has a frame before it
                if frame_time > earliest_next[port]:
                    # definitly should be put in the next layer and not this one
                    next_layer[port] = None
                    logging.warning(f"Skipped frame at port {port}: > earliest_next")
                elif (
                    earliest_next[port] - frame_time < frame_time - latest_current[port]
                ):  # frame time is closer to earliest next than latest current
                    # if it's closer to the earliest next frame than the latest current frame, bump it up
                    # only applying for 2 camera setup where I noticed this was an issue (frames stay out of synch)
                    next_layer[port] = None
                    logging.warning(
                        f"Skipped frame at port {port}: delta < time-latest_current"
                    )
                else:
                    # add the data and increment the index
                    next_layer[port] = self.frame_data.pop(port_index_key)
                    next_layer[port]["sync_index"] = sync_index
                    self.port_current_frame[port] += 1
                    layer_frame_times.append(frame_time)
                    logging.debug(
                        f"Adding to layer from port {port} at index {current_frame_index} and frame time: {frame_time}"
                    )

            logging.debug(f"Unassigned Frames: {len(self.frame_data)}")

            self.mean_frame_times.append(np.mean(layer_frame_times))

            self.current_synched_frames = next_layer

            # notify other processes that the new frames are ready for processing
            # only for tasks that can risk missing frames (i.e. only for gui purposes)
            for q in self.synch_notice_subscribers:
                logging.debug(f"Giving notice of new synched frames packet via {q}")
                q.put("new synched frames available")

            for q in self.synched_frames_subscribers:
                logging.debug(f"Placing new synched frames packet on queue: {q}")
                q.put(self.current_synched_frames)

            sync_index += 1
            self.fps = self.average_fps()

        logging.info("Frame synch worker successfully ended")


if __name__ == "__main__":

    # DON"T DEAL WITH THE SESSION OBJECT IN TESTS...ONLY MORE FOUNDATIONAL ELEMENTS
    from calicam.cameras.camera import Camera
    from calicam.cameras.live_stream import LiveStream
    from calicam.session import Session
    import pandas as pd

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")

    session = Session(config_path)

    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()

    syncr = Synchronizer(session.streams, fps_target=None)

    notification_q = Queue()

    syncr.subscribe_to_notice(notification_q)

    synched_frames = {
        "Sync_Index": [],
        "Port_0_Time": [],
        "Port_1_Time": [],
        "Port_2_Time": [],
    }
    sync_index = 0
    while True:
        synched_frames_notice = notification_q.get()
        synched_frames["Sync_Index"].append(sync_index)
        sync_index += 1

        for port, frame_data in syncr.current_synched_frames.items():

            if frame_data:
                cv2.imshow(f"Port {port}", frame_data["frame"])
                synched_frames[f"Port_{port}_Time"].append(frame_data["frame_time"])
            else:
                synched_frames[f"Port_{port}_Time"].append("dropped")

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    SynchData = pd.DataFrame(synched_frames)
    SynchData.to_csv(Path(config_path, "synch_data.csv"))
