import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread, Event
from queue import Queue
import time

import cv2

# Append main repo to top of path to allow import of backend
from pyxy3d.session import Session
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__

from pyxy3d.gui.widgets import NavigationBarBackNext
config_path = Path(__root__, "tests", "4_cameras_beginning")

session = Session(config_path)
session.load_cameras()
session.load_streams()
syncr = Synchronizer(session.streams, fps_target=5)


notification_q = Queue()
syncr.subscribe_to_notice(notification_q)
logger.info(f"Beginning playback at {time.perf_counter()}")

while not syncr.stop_event.is_set():
    synched_frames_notice = notification_q.get()
    sync_packet = syncr.current_sync_packet
    for port, frame_packet in sync_packet.frame_packets.items():

        if frame_packet:
            cv2.imshow(f"Port {port}", frame_packet.frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        cv2.destroyAllWindows()
        break

    if key == ord("u"):
        syncr.unsubscribe_to_streams()
    
    if key == ord("s"):
        syncr.subscribe_to_streams()

logger.info(f"Playback finished at {time.perf_counter()}")

