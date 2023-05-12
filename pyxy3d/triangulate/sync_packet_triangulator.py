import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
import os
from numba import jit
from numba.typed import Dict, List
import numpy as np
import pandas as pd
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.cameras.synchronizer import Synchronizer, SyncPacket
from queue import Queue
from threading import Thread, Event
from pathlib import Path
from pyxy3d.interface import XYZPacket, Tracker


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
        tracker: Tracker = None,  # used only for getting the point names and tracker name
    ):
        self.camera_array = camera_array
        self.synchronizer = synchronizer
        self.recording_directory = recording_directory

        self.stop_thread = Event()
        self.stop_thread.clear()

        self.tracker = tracker

        self.xyz_history = {
            "sync_index": [],
            "point_id": [],
            "x_coord": [],
            "y_coord": [],
            "z_coord": [],
        }

        self.sync_packet_in_q = Queue(-1)
        self.synchronizer.subscribe_to_sync_packets(self.sync_packet_in_q)

        # assemble numba compatible dictionary
        self.projection_matrices = Dict()
        # self.projection_matrices = {}
        for port, cam in self.camera_array.cameras.items():
            self.projection_matrices[port] = cam.projection_matrix

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
            save_history(self.xyz_history, self.recording_directory, self.tracker)

    def add_packet_to_history(self, xyz_packet: XYZPacket):
        point_count = len(xyz_packet.point_ids)

        if point_count > 0:
            self.xyz_history["sync_index"].extend([xyz_packet.sync_index] * point_count)
            xyz_array = np.array(xyz_packet.point_xyz)
            self.xyz_history["point_id"].extend(xyz_packet.point_ids)
            self.xyz_history["x_coord"].extend(xyz_array[:, 0].tolist())
            self.xyz_history["y_coord"].extend(xyz_array[:, 1].tolist())
            self.xyz_history["z_coord"].extend(xyz_array[:, 2].tolist())


def save_history(
    xyz_history: Dict[str, List], recording_directory: Path, tracker: Tracker = None
):
    

    df_xyz: pd.DataFrame = pd.DataFrame(xyz_history)
    os.makedirs(Path(recording_directory,tracker.name), exist_ok = True)
    df_xyz.to_csv(Path(recording_directory, tracker.name, f"xyz_{tracker.name}.csv"))

    if tracker is not None:
        # save out named data in a tabular format
        df_xyz = df_xyz.rename(
            {
                "x_coord": "x",
                "y_coord": "y",
                "z_coord": "z",
            },
            axis=1,
        )
        df_xyz = df_xyz[["sync_index", "point_id", "x", "y", "z"]]

        df_xyz["point_name"] = df_xyz["point_id"].map(tracker.get_point_name)
        # pivot the DataFrame wider
        df_wide = df_xyz.pivot_table(
            index=["sync_index"], columns="point_name", values=["x", "y", "z"]
        )
        # flatten the column names
        df_wide.columns = ["{}_{}".format(y, x) for x, y in df_wide.columns]
        # reset the index
        df_wide = df_wide.reset_index()
        # merge the rows with the same sync_index
        df_merged = df_wide.groupby("sync_index").agg("first")
        # sort the dataframe
        df_merged = df_merged.sort_index(axis=1, ascending=True)
        df_merged.to_csv(Path(recording_directory, f"tabular_xyz_{tracker.name}.csv"))


# helper function to avoid use of np.unique(return_counts=True) which doesn't work with jit
@jit(nopython=True, cache=True)
def unique_with_counts(arr):
    sorted_arr = np.sort(arr)
    unique_values = [sorted_arr[0]]
    counts = [1]

    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != sorted_arr[i - 1]:
            unique_values.append(sorted_arr[i])
            counts.append(1)
        else:
            counts[-1] += 1

    return np.array(unique_values), np.array(counts)


@jit(nopython=True, parallel=True, cache=True)
def triangulate_sync_index(
    projection_matrices, current_camera_indices, current_point_id, current_img
):
    # sync_indices_xyz = List()
    point_indices_xyz = List()
    obj_xyz = List()

    unique_points, point_counts = unique_with_counts(current_point_id)
    for index in range(len(point_counts)):
        if point_counts[index] > 1:
            # triangulate that point...
            point = unique_points[index]
            points_xy = current_img[current_point_id == point]
            camera_ids = current_camera_indices[current_point_id == point]

            num_cams = len(camera_ids)
            A = np.zeros((num_cams * 2, 4))
            for i in range(num_cams):
                x, y = points_xy[i]
                P = projection_matrices[camera_ids[i]]
                A[(i * 2) : (i * 2 + 1)] = x * P[2] - P[0]
                A[(i * 2 + 1) : (i * 2 + 2)] = y * P[2] - P[1]
            u, s, vh = np.linalg.svd(A, full_matrices=True)
            point_xyzw = vh[-1]
            point_xyz = point_xyzw[:3] / point_xyzw[3]

            point_indices_xyz.append(point)
            obj_xyz.append(point_xyz)

    return point_indices_xyz, obj_xyz
