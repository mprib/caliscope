
import cv2
from pathlib import Path
import time

recordings_path = str(Path("dev", "sample_sessions", "mediapipe_calibration", "port_0.mp4"))
cap = cv2.VideoCapture(recordings_path)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    else:
        # Display the image with the detected hand landmarks
        cv2.imshow("Hand Landmarks", frame)

        # time.sleep(.05)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break