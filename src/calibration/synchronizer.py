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
                    level=logging.INFO)
                    # level=logging.DEBUG)


class Synchronizer:

    def __init__(self, session):
        self.session = session
        
        self.ports = [0,1,2]
        self.synced_frames = []
        self.port_current_frame = [0 for _ in range(len(self.ports))]

        # initialize frame data which will hold everything pushed from 
        # roll_camera() for each port
        self.frame_data = {}
        for port,device in session.rtd.items():
            self.frame_data[port] = []

        logging.info("About to submit Threadpool Harversters")

        # with ThreadPoolExecutor() as executor: 
        #     # executor.submit(self.bundle_frames)

        #     for port, device in session.rtd.items():
        #         executor.submit(self.harvest_corners, device)
        self.threads = []
        for port,device in session.rtd.items():
            t = Thread(target=self.harvest_corners, args=(device,), daemon=True)
            self.threads.append(t)

        for t in self.threads:
            t.start() 

        logging.info("Threadpool harversters just submitted")

        self.bundler = Thread(target= self.bundle_frames, args = ( ), daemon=True)
        self.bundler.start()


        # self.bundler.start()

    # get minimum value of frame_time for next layer
    def earliest_next_frame(self, wait_retry = 0.03, attempt=0):

        try:
            time_of_next_frames = []
            for port in self.ports:
                next_index = self.port_current_frame[port] + 1
                logging.debug(f"about to calculate 'next frame time' for port {port}")
                # port_index_key = str(port) + "_" + str(next_index)
                next_frame_time = self.frame_data[port][next_index]["frame_time"]
                time_of_next_frames.append(next_frame_time)
            return min(time_of_next_frames)

        except IndexError:
            if attempt > 5:
                raise IndexError
            logging.error("Not enough new frames available. Waiting for more frames...")
            time.sleep(wait_retry)
            logging.debug("Reattempting to get earliest next frame")
            self.earliest_next_frame(attempt=attempt+1)

    # def start_corner_harvesters(self):



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
        logging.info("Waiting for frame_data to populate")

        while self.minimum_frame_data() < 10:
            time.sleep(.1)        
            logging.info("Still waiting")

        logging.info("About to start bundling frames...")
        while True:
            time.sleep(.025)
            try:
                cutoff_time = self.earliest_next_frame()
            except IndexError as e:
                logging.info("No frames to process...exiting.")
                print(e)
                break
            
            next_layer = []

            for port in self.ports:
                current_frame = self.port_current_frame[port]
                # frame_data = all_frames[str(port) + "_" + str(port_frame_index)]
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

#TODO: a function that looks at the growing port list and assesses if it is 
# time to call `earliest_next_frame()`

# current_port_frame_indices = [0 for _ in range(len(ports))]

# for key, frame_data in frame_data.items():

#     logging.debug(frame_data)
    # 

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
    # while time.perf_counter - start_time < 30:
        # time.sleep(1)


# # from previous version of synchronizer file
# with open("frame_data.json",) as f:
#     frame_data = json.load(f)