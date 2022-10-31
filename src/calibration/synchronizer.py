# This is a very much work in progress to sort through coding up the basic
# rule for frame syncing that I have in mind. Put all the frames in the first
# index. If one frame was read after the earliest frame in the next index,
# move it to the next index

import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread
from multiprocessing import Process

import cv2
import numpy as np

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.calibration.mono_calibrator import MonoCalibrator
from src.cameras.camera import Camera
from src.session import Session

logging.basicConfig(filename="synchronizer.log", 
                    filemode = "w", 
                    # level=logging.INFO)
                    level=logging.DEBUG)


class Synchronizer:

    def __init__(self, session):
        self.session = session
        
        self.ports = []
        self.synced_frames = []

        # initialize frame data which will hold everything pushed from 
        # roll_camera() for each port
        self.frame_data = {}
        for port,device in session.rtd.items():
            self.ports.append(port)
            self.frame_data[port] = []
        self.port_current_frame = [0 for _ in range(len(self.ports))]

        logging.info("About to submit Threadpool Harversters")

        self.shutter_sync = Queue()

        self.threads = []
        for port,device in session.rtd.items():
            device.assign_shutter_sync(self.shutter_sync)

            t = Thread(target=self.harvest_corners, args=(device,), daemon=True)
            self.threads.append(t)

        for t in self.threads:
            t.start()
        
        self.bundler = Thread(target= self.bundle_frames, args = ())
        self.bundler.start()

        logging.info("Threadpool harversters just submitted")



        # self.bundler.start()

    # get minimum value of frame_time for next layer
    def earliest_next_frame(self, wait_retry = 0.05, attempt=0):

        time_of_next_frames = []
        for port in self.ports:
            next_index = self.port_current_frame[port] + 1
            # logging.debug(f"about to calculate 'next frame time' for port {port}")
            # port_index_key = str(port) + "_" + str(next_index)
            next_frame_time = self.frame_data[port][next_index]["frame_time"]
            # logging.debug(f"'next frame time' for port {port} is {next_frame_time}")
            time_of_next_frames.append(next_frame_time)

            if time_of_next_frames is None:
                logging.debug("Next frame time None. Investigate.")
        return min(time_of_next_frames)

    def frame_slack(self):
        """Determine how many unassigned frames are sitting in self.dataframe"""
        
        frame_count = {port:len(data) for port, data in self.frame_data.items()}

        slack = [frame_count[port] - self.port_current_frame[port] for port in self.ports] 
        logging.debug(f"Slack in frames is {slack}")  
        return min(slack) 
        

    def harvest_corners(self, device):
        device.push_to_reel = True
        port = device.cam.port
        device.track_charuco = True
        
        logging.info(f"Beginning to harvest corners at port {port}")
        frame_index = 0

        while True:

            frame_time, frame, corner_ids, frame_corners, board_FOR_corners = device.reel.get()
            print(f"Reading from port {port}: frame created at {round(frame_time,3)} read at {round(time.perf_counter(),3)}; ")                
            
            # frame_time = frame_time.tolist()
            corner_ids = corner_ids.tolist()
            frame_corners = frame_corners.tolist()
            board_FOR_corners = board_FOR_corners.tolist()

            self.frame_data[port].append(
                {
                    "port": port,
                    "frame_index": frame_index,
                    "frame_time": frame_time,
                    "corner_ids": corner_ids,
                    "frame_corners": frame_corners,
                    "board_FOR_corners": board_FOR_corners # corner location in board frame of reference
                })
            
            frame_index += 1
            # note: this imshow is more for debugging...not sure what to do with it
            cv2.putText(frame,f"{round(device.FPS_actual,1)} FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
            cv2.putText(frame, f"Time: {round(frame_time,1)}", (30, 70),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
            cv2.imshow(f"Port {port}", frame)  # imshow is still IO, so threading may remain best choice     
         
            key = cv2.waitKey(1)

            if key == ord("q"):
                device.track_charuco = False
                break


    def minimum_frame_data(self):
        min_frames = 0
        for port in self.ports:
            frame_count = len(self.frame_data[port])
            if frame_count > min_frames:
                min_frames = frame_count

        return min_frames
                
    def bundle_frames(self):
        
        # need to have 2 frames to assess bundling
        for port in self.ports:
            self.shutter_sync.put("fire")

        logging.info("About to start bundling frames...")
        while True:
            # Trigger device to proceed with reading frame and pushing to reel
            for port in self.ports:
                self.shutter_sync.put("fire")
            
            # sleep function to throttle frame rate
            # time.sleep(.2)

            # wait for frame data to populate
            while self.frame_slack() < 2:
                time.sleep(.01)


            cutoff_time = self.earliest_next_frame()
            next_layer = []
            # test call to see what it returns

            for port in self.ports:
                current_frame = self.port_current_frame[port]
                current_frame_data = self.frame_data[port][current_frame]
                frame_time = current_frame_data["frame_time"]

                # placeholder here is where the actual corner data would go
                placeholder = f"{port}_{current_frame}_{frame_time}"

                if frame_time < cutoff_time:
                    #add the data and increment the index
                    next_layer.append(placeholder)      
                    self.port_current_frame[port] +=1
                else:
                    next_layer.append(None)

            logging.debug(f"Next Layer: {next_layer}")

            self.synced_frames.append(next_layer)

    def log_synced_frames(self):

        for i in range(len(self.synced_frames)):
            logging.INFO(f"Synced: {self.synced_frames[i]}")


if __name__ == "__main__":

    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    session.load_cameras()
    session.find_additional_cameras() # looking to add a third
    start_time = time.perf_counter()

    session.load_rtds()

    syncr = Synchronizer(session)
    # print(syncr.synced_frames)

    for i in range(len(syncr.synced_frames)):
        logging.info(f"From __main__: {syncr.synced_frames[i]}")
