import logging

logging.basicConfig(filename="dispatcher.log", filemode="w", level=logging.INFO)
# level=logging.DEBUG)

import sys
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np

# Append main repo to top of path to allow import of backend

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.synchronizer import Synchronizer


class Dispatcher:
    """This class pulls the frame bundle from the synced_frames_q of
    the syncronizer and then partitions the frames out to various queues.
    Callers of the dispather can add a queu to it along with its associated
    port, and the dispatcher will start populating that que.

    if the port is not a single integer, but instead contains an _
    (such as 0_1) then a list of synced frame pairs will be put on the queue"""

    def __init__(self, synchronizer):
        self.synchronizer = synchronizer

        self.queues = defaultdict(
            list
        )  # to be populated with tuples of form (port, queue)

        self.run_dispatch = True
        self.thread = Thread(target=self.dispatch_frames_worker, args=[], daemon=True)
        logging.info("Starting dispatcher thread")
        self.thread.start()

    def dispatch_frames_worker(self):
        logging.info("spinning up dispatch worker")
        while self.run_dispatch:
            logging.debug("re-entering dispatch loop again")
            frame_bundle = self.synchronizer.synced_frames_q.get()

            for ports, q_list in self.queues.items():
                logging.debug(f"Port(s) {ports} has list of {q_list}")

                # push single frame to the mono queues
                if type(ports) == int:
                    for q in q_list:
                        q.put(frame_bundle[ports]["frame"])

                # push synched stereo pairs for stereo queues
                if type(ports) == tuple:
                    for q in q_list:
                        stereo_bundle = []
                        stereo_bundle.append(frame_bundle[ports[0]]["frame"])
                        stereo_bundle.append(frame_bundle[ports[1]]["frame"])
                        q.put(stereo_bundle)

    def add_mono_queue(self, port, q):
        logging.info(f"Adding queue for port {port}")
        self.queues[port].append(q)
        logging.info(f"All Queues: {self.queues}")
        logging.info(f"Successfully added queue for port {port}")

    def add_stereo_queue(self, stereo_ports, q):
        logging.info(f"Adding queue for ports {stereo_ports[0]}")
        self.queues[stereo_ports].append(q)
        logging.info(f"All Queues: {self.queues}")
        logging.info(f"Successfully added queue for port {stereo_ports}")


if __name__ == "__main__":
    from src.session import Session

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "default_session")
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    syncr = Synchronizer(session.streams, fps_target=6)

    logging.info("Building dispatcher")
    dispatchr = Dispatcher(syncr)

    mono_q = Queue()
    mono_port = 1
    logging.info("Adding test queue")
    dispatchr.add_mono_queue(mono_port, mono_q)

    stereo_q = Queue()
    stereo_ports = (0, 2)
    logging.info("Adding test stereo queue")
    dispatchr.add_stereo_queue(stereo_ports, stereo_q)

    while True:
        logging.debug("About to get frame from test queue")
        frame = mono_q.get()
        cv2.imshow(f"Port {stereo_ports}", frame)

        key = cv2.waitKey(1)
        frames = stereo_q.get()

        cv2.imshow(f"Port {stereo_ports[0]}", frames[0])
        cv2.imshow(f"Port {stereo_ports[1]}", frames[1])

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
