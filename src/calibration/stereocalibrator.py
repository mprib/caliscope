 # I feel like I've forgotten how to program. OK. what the heck and I doing here?



from threading import Thread
import cv2
import time
import sys

import numpy as np
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
    session.find_additional_cameras() 

    session.load_rtds()
    # session.adjust_resolutions()

    print(session.rtd)
    while True:
        print("looping")
        for port, device in session.rtd.items():
            corner_count = len(device.mono_cal.frame_corners)
            id_count = len(device.mono_cal.frame_corner_ids)

            if corner_count > 0 and device.frame_stereo_read == False:
                device.frame_stereo_read = True
                print(f"Camera {port}: frame read at {device.frame_time}")                
                print(device.mono_cal.frame_corner_ids)
                print(device.mono_cal.frame_corners)

            cv2.putText(device.frame,f"{round(device.FPS_actual,1)} FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), 3)
            cv2.imshow(f"Port {port}", device.frame)       
         
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        if key == ord("c"):
            for port, device in session.rtd.items():
                device.charuco_being_tracked = not device.charuco_being_tracked 

        # time.sleep(.03)           

    