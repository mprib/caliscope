import numpy as np
import pandas as pd
from caliscope.cameras.camera_array import CameraArray
from caliscope.cameras.synchronizer import Synchronizer, SyncPacket
from queue import Queue
from threading import Thread, Event
from pathlib import Path
from caliscope.packets import XYZPacket

from caliscope.triangulate.triangulation import triangulate_sync_index
import caliscope.logger

logger = caliscope.logger.get(__name__)

class SyncPacketTriangulator:
    """
    Will place 3d packets on subscribed queues and save consolidated data in csv
    format to output_path if provided
    """

    def __init__(
        self,
        camera_array: CameraArray,
        synchronizer: Synchronizer,
        recording_directory: Path = None,
        tracker_name:str = None,  # used only for getting the point names and tracker name
    ):
        self.camera_array = camera_array
        self.synchronizer = synchronizer
        self.recording_directory = recording_directory

        self.stop_thread = Event()
        self.stop_thread.clear()

        self.tracker_name = tracker_name

        self.xyz_history = {
            "sync_index": [],
            "point_id": [],
            "x_coord": [],
            "y_coord": [],
            "z_coord": [],
        }

        self.sync_packet_in_q = Queue(-1)
        self.synchronizer.subscribe_to_sync_packets(self.sync_packet_in_q)

        self.projection_matrices = self.camera_array.projection_matrices
        
        self.subscribers = []
        self.running = True
        self.thread = Thread(target=self.process_incoming, args=(), daemon=True)
        self.thread.start()

    def subscribe(self, queue: Queue):
        self.subscribers.append(queue)

    def unsubscriber(self, queue: Queue):
        self.subscribers.remove(queue)

    def process_incoming(self):
        # waiting to set running property here was causing issues with identifying state of thread.
        # set property to true then start thread...

        # self.running = True
        while not self.stop_thread.is_set():
            sync_packet: SyncPacket = self.sync_packet_in_q.get()

            if sync_packet is None:
                # No more sync packets after this... wind down
                self.stop_thread.set()
                logger.info(
                    "End processing of incoming sync packets...end signaled with `None` packet"
                )
            else:
                logger.debug(
                    f"Sync Packet {sync_packet.sync_index} acquired with {sync_packet.frame_packet_count} frames"
                )
                # only attempt to process if data exists
                if sync_packet.frame_packet_count >= 2:
                    cameras, point_ids, imgs_xy = sync_packet.triangulation_inputs
                    logger.debug("Attempting to triangulate synced frames")
                    # prepare for jit
                    cameras = np.array(cameras)
                    point_ids = np.array(point_ids)
                    imgs_xy = np.array(imgs_xy)

                    logger.debug(f"Cameras are {cameras} and point_ids are {point_ids}")
                    if len(np.unique(cameras)) >= 2:
                        logger.debug(f"Points observed on cameras {np.unique(cameras)}")
                        point_id_xyz, points_xyz = triangulate_sync_index(
                            self.projection_matrices, cameras, point_ids, imgs_xy
                        )

                        logger.debug(
                            f"Synch Packet {sync_packet.sync_index} | Point ID: {point_id_xyz} | xyz: {points_xyz}"
                        )

                        xyz_packet = XYZPacket(
                            sync_packet.sync_index, point_id_xyz, points_xyz
                        )
                        logger.info(
                            f"Placing xyz pacKet for index {sync_packet.sync_index} with {len(xyz_packet.point_ids)} points"
                        )
                        for q in self.subscribers:
                            q.put(xyz_packet)

                        # if self.output_path is not None:
                        self.add_packet_to_history(xyz_packet)

        self.running = False

        if self.recording_directory is not None:
            logger.info(f"Saving xyz point data to {self.recording_directory}")
            self.save_history()

    def add_packet_to_history(self, xyz_packet: XYZPacket):
        point_count = len(xyz_packet.point_ids)

        if point_count > 0:
            self.xyz_history["sync_index"].extend([xyz_packet.sync_index] * point_count)
            self.xyz_history["point_id"].extend(xyz_packet.point_ids)

            xyz_array = np.array(xyz_packet.point_xyz)
            self.xyz_history["x_coord"].extend(xyz_array[:, 0].tolist())
            self.xyz_history["y_coord"].extend(xyz_array[:, 1].tolist())
            self.xyz_history["z_coord"].extend(xyz_array[:, 2].tolist())
            
    

    def save_history(self)->None:
        """
        If a recording directory is provided, then save the xyz directory into it
        If a tracker name is provided, then base name on the tracker name
        """
        df_xyz: pd.DataFrame = pd.DataFrame(self.xyz_history)
       
        if self.recording_directory is not None: 
            if self.tracker_name is None:
                filename = "xyz.csv"
            else:
                filename = f"xyz_{self.tracker_name}.csv"
                df_xyz.to_csv(Path(self.recording_directory,filename))

