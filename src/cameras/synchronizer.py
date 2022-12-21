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

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class Synchronizer:
    def __init__(self, streams: dict, fps_target):
        self.streams = streams
        self.current_bundle = None

        self.notice_subscribers = []  # queues that will be notified of new bundles
        self.bundle_subscribers = []    # queues that will receive actual frame data
        
        self.frame_data = {}
        self.stop_event = Event()

        self.ports = []
        for port, stream in self.streams.items():
            self.ports.append(port)

        self.fps_target = fps_target
        if fps_target is not None:
            self.fps = fps_target

        self.initialize_ledgers()
        self.spin_up() 

    def stop(self):
        self.stop_event.set()
        self.bundler.join()
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

        logging.info("Starting frame bundler...")
        self.bundler = Thread(target=self.bundle_frames, args=(), daemon=True)
        self.bundler.start()
        
    def subscribe_to_notice(self, q):
        # subscribers are notified via the queue that a new frame bundle is available
        # this is intended to avoid issues with latency due to multiple iterations
        # of frames being passed from one queue to another
        logging.info("Adding queue to receive notice of bundle update")
        self.notice_subscribers.append(q)

    def subscribe_to_bundle(self, q):
        logging.info("Adding queue to receive frame bundle")
        self.bundle_subscribers.append(q)

    def release_bundle_q(self,q):
        logging.info("Releasing record queue")
        self.bundle_subscribers.remove(q)

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

            if frame_time == -1: # signal from recorded stream that end of file reached
                break
            # once toggled, keep pushing the poison pill
            
            self.frame_data[f"{port}_{frame_index}"] = {
                "port": port,
                "frame": frame,
                "frame_index": frame_index,
                "frame_time": frame_time,
            }

            logging.debug(f"Frame data harvested from reel {port} with index {frame_index} and frame time of {frame_time}")
            self.port_frame_count[port] += 1

        logging.info(f"Frame harvester for port {port} completed")

    # get minimum value of frame_time for next layer
    def earliest_next_frame(self, port):
        """Looks at next unassigned frame across the ports to determine
        the earliest time at which each of them was read"""
        times_of_next_frames = []
        for p in self.ports:
            next_index = self.port_current_frame[p] + 1
            frame_data_key =  f"{p}_{next_index}"
            
            # problem with outpacing the threads reading data in, so wait if need be
            while frame_data_key not in self.frame_data.keys():
                logging.debug(f"Waiting in a loop for frame data to populate with key: {frame_data_key}")
                time.sleep(.001)

            next_frame_time = self.frame_data[frame_data_key]["frame_time"]
            if p != port:
                times_of_next_frames.append(next_frame_time)

        return min(times_of_next_frames)
    
    def latest_current_frame(self, port):
        """Provides the latest frame_time of the current frames not inclusive of the provided port """
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
        """"""
        # only look at the most recent layers
        if len(self.mean_frame_times) > 10:
            self.mean_frame_times = self.mean_frame_times[-10:]

        delta_t = np.diff(self.mean_frame_times)
        mean_delta_t = np.mean(delta_t)

        return 1 / mean_delta_t

    def bundle_frames(self):

        logging.info(f"Waiting for all ports to begin harvesting corners...")

        # need to have 2 frames to assess bundling
        for port in self.ports:
            self.streams[port].shutter_sync.put("fire")
            self.streams[port].shutter_sync.put("fire")


        sync_time = time.perf_counter()

        logging.info("About to start bundling frames...")
        while not self.stop_event.is_set():

            # Enforce a wait period to hit target FPS, unless you have excess slack
            if self.frame_slack() < 2:
                # Trigger device to proceed with reading frame and pushing to reel
                if self.fps_target is not None:
                    wait_time = 1 / self.fps_target
                    while time.perf_counter() < sync_time + wait_time:
                        time.sleep(0.001)

                sync_time = time.perf_counter()
                for port in self.ports:
                    self.streams[port].shutter_sync.put("fire")

            next_layer = {}
            layer_frame_times = []
            
            # build earliest next/latest current dictionaries for each port to determine where to put frames           
            # must be done before going in and making any updates to the frame index
            earliest_next = {}
            latest_current = {}
            
            # the two dictionaries below are for debugging purposes
            # frame_time_current = {}
            # frame_time_next = {}

            for port in self.ports:
                earliest_next[port] = self.earliest_next_frame(port)
                latest_current[port] = self.latest_current_frame(port)
                current_frame_index = self.port_current_frame[port]
                
                # inserting the dictionaries below to debug issue with all frames dropping
                # port_index_key = f"{port}_{current_frame_index}"
                # current_frame_data = self.frame_data[port_index_key]
                # frame_time_current[port] = current_frame_data["frame_time"]
                # port_next_index_key = f"{port}_{current_frame_index+1}"
                # next_frame_data = self.frame_data[port_next_index_key]
                # frame_time_next[port] = next_frame_data["frame_time"]
                
            for port in self.ports:
                current_frame_index = self.port_current_frame[port]

                port_index_key = f"{port}_{current_frame_index}"
                current_frame_data = self.frame_data[port_index_key]
                frame_time = current_frame_data["frame_time"]

                # don't put a frame in a bundle if the next bundle has a frame before it
                if frame_time > earliest_next[port]:
                    # definitly should be put in the next layer and not this one
                    next_layer[port] = None
                    logging.warning(f"Skipped frame at port {port}: > earliest_next")
                elif earliest_next[port] - frame_time < frame_time-latest_current[port]: # frame time is closer to earliest next than latest current
                    # if it's closer to the earliest next frame than the latest current frame, bump it up
                    # only applying for 2 camera setup where I noticed this was an issue (frames stay out of synch)
                    next_layer[port] = None
                    logging.warning(f"Skipped frame at port {port}: delta < time-latest_current")
                else:
                    # add the data and increment the index
                    next_layer[port] = self.frame_data.pop(port_index_key)
                    self.port_current_frame[port] += 1
                    layer_frame_times.append(frame_time)
                    logging.debug(f"Adding to layer from port {port} at index {current_frame_index} and frame time: {frame_time}")
                    
            logging.debug(f"Unassigned Frames: {len(self.frame_data)}")

            self.mean_frame_times.append(np.mean(layer_frame_times))

            self.current_bundle = next_layer
            # notify other processes that the current bundle is ready for processing
            # only for tasks that can risk missing a frame bundle
            for q in self.notice_subscribers:
                logging.debug(f"Giving notice of new bundle via {q}")
                q.put("new bundle available")

            for q in self.bundle_subscribers:
                logging.debug(f"Placing new bundle on queue: {q}")
                logging.debug("Placing bundle on subscribers queue")
                q.put(self.current_bundle)

            self.fps = self.average_fps()

        logging.info("Frame bundler successfully ended")

if __name__ == "__main__":

    # DON"T DEAL WITH THE SESSION OBJECT IN TESTS...ONLY MORE FOUNDATIONAL ELEMENTS
    from src.cameras.camera import Camera
    from src.cameras.live_stream import LiveStream
    from src.session import Session
    import pandas as pd

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")

    session = Session(config_path)

    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()

    # cameras = []
    # ports = [0, 1, 2]

    # for port in ports:
    #     cameras.append(Camera(port))

    # streams = {}
    # for cam in cameras:
    #     streams[cam.port] = LiveStream(cam)

    syncr = Synchronizer(session.streams, fps_target=None)

    notification_q = Queue()

    syncr.subscribe_to_notice(notification_q)
    
    bundle_data = {"Bundle":[],
                   "Port_0_Time":[],
                   "Port_1_Time":[],
                   "Port_2_Time":[]}
    bundle_index = 0
    while True:
        frame_bundle_notice = notification_q.get()
        bundle_data["Bundle"].append(bundle_index)
        bundle_index += 1

        for port, frame_data in syncr.current_bundle.items():
            
            if frame_data:
                cv2.imshow(f"Port {port}", frame_data["frame"])
                bundle_data[f"Port_{port}_Time"].append(frame_data["frame_time"])
            else:
                bundle_data[f"Port_{port}_Time"].append("dropped")
                
        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    SynchData = pd.DataFrame(bundle_data)
    SynchData.to_csv(Path(config_path,"synch_data.csv"))
    # print(bundle_data) 