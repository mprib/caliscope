
import cv2
from pathlib import Path
import time
from pyxy3d.trackers.hand_tracker import HandTracker
from pyxy3d.interface import PointPacket, FramePacket

from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.interface import PointPacket, FramePacket, SyncPacket
from pyxy3d.triangulate.real_time_triangulator import RealTimeTriangulator
from pyxy3d.cameras.camera_array import CameraArray, CameraData, get_camera_array
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d.calibration.charuco import Charuco, get_charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.configurator import Configurator

port = 0
session_path = str(Path("dev", "sample_sessions", "mediapipe_calibration", "port_0.mp4"))
hand_tracker = HandTracker()

cap = cv2.VideoCapture(session_path)

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