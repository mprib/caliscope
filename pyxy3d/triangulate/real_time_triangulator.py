

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from numba import jit
from numba.typed import Dict, List
import numpy as np
import pandas as pd
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.cameras.synchronizer import Synchronizer, SyncPacket
from queue import Queue
from threading import Thread, Event
from pathlib import Path
from pyxy3d.interface import XYZPacket


class RealTimeTriangulator:
    """
    Will place 3d packets on subscribed queues and save consolidated data in csv
    format to output_path if provided
    """    
    def __init__(self,camera_array:CameraArray, synchronizer:Synchronizer, output_directory:Path=None):
        self.camera_array = camera_array
        self.synchronizer = synchronizer
        self.output_directory = output_directory

        self.stop_thread = Event()
        self.stop_thread.clear()
        # self._sync_packet_history = []     
        self.xyz_history = []
        self.sync_packet_in_q = Queue(-1) 
        self.synchronizer.subscribe_to_sync_packets(self.sync_packet_in_q)
        
        # assemble numba compatible dictionary
        self.projection_matrices = Dict() 
        # self.projection_matrices = {}
        for port, cam in self.camera_array.cameras.items():
            self.projection_matrices[port] = cam.projection_matrix

        self.subscribers = []
        self.thread = Thread(target=self.process_incoming, args=(), daemon=True)
        self.thread.start()
        
    def subscribe(self, queue:Queue):
        self.subscribers.append(queue)
        
    def unsubscriber(self,queue:Queue):
        self.subscribers.remove(queue)


    def process_incoming(self):

        self.running = True
        while not self.stop_thread.is_set():

            sync_packet:SyncPacket = self.sync_packet_in_q.get()

            if sync_packet is None:
                # No more sync packets after this... wind down
                self.stop_thread.set()
                logger.info("End processing of incoming sync packets...end signaled with `None` packet")
            else:    
                logger.info(f"Sync Packet {sync_packet.sync_index} acquired...")     
                # self._sync_packet_history.append(sync_packet)
    
                cameras, point_ids, imgs_xy = sync_packet.triangulation_inputs

                # only attempt to process if data exists
                if len(cameras) >= 2:
                    logger.info("Attempting to triangulate synced frames")
                    # prepare for jit
                    cameras = np.array(cameras)
                    point_ids = np.array(point_ids)
                    imgs_xy = np.array(imgs_xy)

                    logger.info(f"Cameras are {cameras} and point_ids are {point_ids}")
                    
                    point_id_xyz, points_xyz = triangulate_sync_index(
                        self.projection_matrices, cameras, point_ids, imgs_xy
                    )
            
                    logger.info(f"Synch Packet {sync_packet.sync_index} | Point ID: {point_id_xyz} | xyz: {points_xyz}")

                    xyz_packet = XYZPacket(sync_packet.sync_index,point_id_xyz,points_xyz)
                    for q in self.subscribers:
                        q.put(xyz_packet)
                    
                    # if self.output_path is not None:
                    self.xyz_history.append(xyz_packet)
                              
        self.running = False 

        if self.output_directory is not None:
            logger.info(f"Saving xyz point data to {self.output_directory}")
            self.save_history()
            
    def save_history(self):
        
        xyz_history = {"sync_index":[], 
                        "point_id":[], 
                        "x_coord":[],
                        "y_coord":[], 
                        "z_coord":[]}

        for packet in self.xyz_history:
            point_count = len(packet.point_ids)
            if point_count>0:
                xyz_history["sync_index"].extend([packet.sync_index]*point_count)
                xyz_array = np.array(packet.point_xyz)
                xyz_history["point_id"].extend(packet.point_ids)
                xyz_history["x_coord"].extend(xyz_array[:,0].tolist())
                xyz_history["y_coord"].extend(xyz_array[:,1].tolist())
                xyz_history["z_coord"].extend(xyz_array[:,2].tolist())
       
        xyz_history:pd.DataFrame = pd.DataFrame(xyz_history)
        xyz_history.to_csv(Path(self.output_directory,"xyz_history.csv"))
        
# helper function to avoid use of np.unique(return_counts=True) which doesn't work with jit
@jit(nopython=True)
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

# NOTE: jit does not appear to improve processing time even after first compilation.
# Test difference in the future with more points...
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