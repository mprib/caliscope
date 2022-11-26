import logging

# LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.DEBUG
LOG_FILE = "dispatcher.log"
logging.basicConfig(filename=LOG_FILE, filemode="w", level=LOG_LEVEL)
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
    Callers of the dispather can add a queue to it along with its associated
    ports, and the dispatcher will start populating that queue with a list of
    frames associated with those ports.

    ports argument can be either an integer or a tuple
    """

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
                logging.debug(f"Port(s) {ports} has list of {len(q_list)} items long")

                # ensure that ports is a list, even if only one
                if type(ports) == int:
                    ports = [ports]

                for q in q_list:
                    frames = []
                    for port in ports:
                        # A list of frames in port order is placed on the queue
                        # must be accessed downstream by list index
                        frames.append(frame_bundle[port]["frame"])
                    q.put(frames)

    def add_queue(self, port, q):
        logging.info(f"Adding queue for port(s) {port}")
        self.queues[port].append(q)
        logging.info(f"All Queues: {self.queues}")
        logging.info(f"Successfully added queue for port {port}")


if __name__ == "__main__":
    from src.session import Session

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "default_session")
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    syncr = Synchronizer(session.streams, fps_target=12)

    logging.info("Building dispatcher")
    dispatchr = Dispatcher(syncr)

    stereo_q = Queue()
    stereo_ports = (0, 2)
    logging.info("Adding test stereo queue")
    dispatchr.add_queue(stereo_ports, stereo_q)

    mono_q = Queue()
    mono_port = 1
    logging.info("Adding test queue")
    dispatchr.add_queue(mono_port, mono_q)

    while True:
        logging.debug("About to get frame from test queue")
        frame = mono_q.get()
        frames = stereo_q.get()

        cv2.imshow(f"Port {mono_port}", frame[0])

        cv2.imshow(f"Port {stereo_ports[0]}", frames[0])
        cv2.imshow(f"Port {stereo_ports[1]}", frames[1])

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
