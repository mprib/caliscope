import logging

logging.basicConfig(
    filename="dispatcher.log",
    filemode="w",
    # level=logging.INFO)
    level=logging.DEBUG,
)

import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np

# Append main repo to top of path to allow import of backend

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.synchronizer import Synchronizer


class Dispatcher:
    synchronizer: Synchronizer

    def __init__(self, synchronizer):
        self.synchronizer = synchronizer
        self.queues = []  # to be populated with tuples of form (port, queue)

        self.run_dispatch = True
        self.thread = Thread(target=self.dispatch_frames_worker, args=[], daemon=True)
        logging.info("About to run thread")
        self.thread.run()

    def dispatch_frames_worker(self):
        logging.info("spinning up dispatch worker")
        while self.run_dispatch:
            logging.debug("re-entering dispatch loop again")
            print(self.queues)
            frame_bundle = self.synchronizer.synced_frames_q.get()
            # logging.debug(frame_bundle)
            for port, q in self.queues:
                q.put(frame_bundle[port]["frame"])

    def add_queue(self, port, q):
        logging.info(f"Adding queue for port {port}")
        self.queues.append((port, q))
        logging.info(f"Successfully added queue for port {port}")


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

    test_q = Queue()
    port = 1
    logging.debug("Adding test queue")
    dispatchr.add_queue(port, test_q)

    while True:
        logging.debug("About to get frame from test queue")
        frame = test_q.get()
        cv2.imshow(f"Port {port}", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break
