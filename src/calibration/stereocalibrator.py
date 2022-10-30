

from queue import Queue
from threading import Thread
import cv2
import time
import sys
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# Append main repo to top of path to allow import of backend
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera
from src.calibration.mono_calibrator import MonoCalibrator
from src.calibration.charuco import Charuco

from src.session import Session


    

if __name__ == "__main__":

    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    session.load_cameras()
    session.find_additional_cameras() # looking to add a third
    start_time = time.perf_counter()
    corner_queue = Queue() # a place for threads to push corner data...will be converted to dict and ultimately json for development

    end_harvest = False
    session.load_rtds()
    # session.adjust_resolutions()


    def harvest_corners(device, corner_queue):
        device.push_to_reel = True
        port = device.cam.port
        device.track_charuco = True
        
        frame_index = 0

        while True:
            # print(device.mono_cal.frame_corner_ids)
            # print(device.mono_cal.frame_corners)

            frame_time, frame, corner_ids, frame_corners, board_FOR_corners = device.reel.get()
            print(f"Reading from port {port}: frame created at {round(frame_time,3)} read at {round(time.perf_counter(),3)}; ")                
            
            # frame_time = frame_time.tolist()
            corner_ids = corner_ids.tolist()
            frame_corners = frame_corners.tolist()
            board_FOR_corners = board_FOR_corners.tolist()

            corner_queue.put(
                {
                    "port": port,
                    "frame_index": frame_index,
                    "frame_time": frame_time,
                    "corner_ids": corner_ids,
                    "frame_corners": frame_corners,
                    "board_FOR_corners": board_FOR_corners # corner location in board frame of reference
                })
            
            frame_index += 1
            # print(f"Corner IDs: {corner_ids}")
            # print(f"Image Corner Loc: {frame_corners}")
            # print(f"Board FOR Corner Loc: {board_FOR_corners}")
            cv2.putText(frame,f"{round(device.FPS_actual,1)} FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
            cv2.putText(frame, f"Time: {round(frame_time,1)}", (30, 70),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
            cv2.imshow(f"Port {port}", frame)  # imshow is still IO, so threading may remain best choice     
         
            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    

    with ThreadPoolExecutor() as executor: 

        for port, device in session.rtd.items():
            executor.submit(harvest_corners, device, corner_queue)



    
    # now move across the corner_queue in a sequential way

    all_frames = {}
    for _ in range(corner_queue.qsize()):
        port_time_corners = corner_queue.get()
        port = port_time_corners["port"]
        frame_time = port_time_corners["frame_time"]
        frame_index = port_time_corners["frame_index"]
        frame_key = f"{port}_{frame_index}"

        all_frames[frame_key] = port_time_corners

#%%
    with open("all_frames.json","w") as f:
        json.dump(all_frames, f)