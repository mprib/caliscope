 # I feel like I've forgotten how to program. OK. what the heck and I doing here?



from threading import Thread
import cv2
import time
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# Append main repo to top of path to allow import of backend
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera
from src.calibration.mono_calibrator import MonoCalibrator
from src.calibration.charuco import Charuco

from src.session import Session


# def harvest_frame_time_corners(device):
#     device.push_to_reel = True
#     port = device.cam.port

#     while True:
#         print(f"harvesting from port {port}")
#         frame_time, frame = device.reel.get()
#         frame_time = round(frame_time,1)
#         cv2.putText(frame,f"{round(device.FPS_actual,1)} FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
#         cv2.putText(frame, f"Time: {frame_time}", (30, 100),cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,250), 3)
#         cv2.imshow(f"Port {port}", frame)       

#         if device.push_to_reel == False:
#             print(f"End Push to reel for device at port {port}")
#             break

    

if __name__ == "__main__":

    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    session.load_cameras()
    # session.find_additional_cameras() 

    end_harvest = False
    session.load_rtds()
    # session.adjust_resolutions()


    def harvest_corners(device):
        device.push_to_reel = True
        port = device.cam.port
        device.
        while True:
            print(f"Camera {port}: frame read at {device.frame_time}")                
            # print(device.mono_cal.frame_corner_ids)
            # print(device.mono_cal.frame_corners)

            frame_time, frame = device.reel.get()

            cv2.putText(frame,f"{round(device.FPS_actual,1)} FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
            cv2.putText(frame, f"Time: {round(frame_time,1)}", (30, 70),cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
            cv2.imshow(f"Port {port}", frame)  # imshow is still IO, so threading may remain best choice     
         
            key = cv2.waitKey(1)

            if key == ord("q"):
                break


    with ThreadPoolExecutor() as executor: 

        for port, device in session.rtd.items():
            executor.submit(harvest_corners, device)

    # while True:
# 
        # key = cv2.waitKey(1)
        # if key == ord("q"):
            # break


    # while True:
    #     key = cv2.waitKey(1)
    #     if key == ord("q"):
    #         for port, device in session.rtd.items():
    #             device.push_to_reel = False


    # for port, device in session.rtd.items():
        # device.push_to_reel = True
    # print(session.rtd)
    # while True:
        # print("looping")
        # for port, device in session.rtd.items():
# 
            # corner_count = len(device.mono_cal._frame_corners)
            # id_count = len(device.mono_cal._frame_corner_ids)
# 
            # if corner_count > 0 and device.frame_stereo_read == False:
                # device.frame_stereo_read = True
                # print(f"Camera {port}: frame read at {device.frame_time}")                
                # print(device.mono_cal.frame_corner_ids)
                # print(device.mono_cal.frame_corners)
            # frame_time, frame = device.reel.get()
# 
            # cv2.putText(frame,f"{round(device.FPS_actual,1)} FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
            # cv2.putText(frame, f"Time: {frame_time}", (30, 100),cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,250), 3)
            # cv2.imshow(f"Port {port}", frame)  # imshow is still IO, so threading may remain best choice     
        #  
        # key = cv2.waitKey(1)
# 
        # if key == ord("q"):
            # break
# 
        # if key == ord("c"):
            # for port, device in session.rtd.items():
                # device.charuco_being_tracked = not device.charuco_being_tracked 

        # time.sleep(.03)           

    