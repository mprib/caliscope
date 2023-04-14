
import cv2
from pathlib import Path
import time
from pyxy3d.trackers.hand_tracker import HandTracker
from pyxy3d.interface import PointPacket, FramePacket

port = 0
recordings_path = str(Path("dev", "sample_sessions", "mediapipe_calibration", "port_0.mp4"))
cap = cv2.VideoCapture(recordings_path)
hand_tracker = HandTracker()

while True:
    success, frame = cap.read()
    if not success:
        break
    
    else:
        # Display the image with the detected hand landmarks
        hand_points_packet:PointPacket = hand_tracker.get_points(frame)

        frame_packet = FramePacket(port,time.time(),frame,points=hand_points_packet)
        # cv2.imshow("Hand Landmarks", frame_packet.frame_with_points)
        cv2.imshow("Hand Landmarks", frame_packet.frame_with_points)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        
cv2.destroyAllWindows()