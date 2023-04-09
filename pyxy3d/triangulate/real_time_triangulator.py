

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from numba import jit
from numba.typed import Dict, List
import numpy as np
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.cameras.synchronizer import Synchronizer, SyncPacket
from queue import Queue
from threading import Thread, Event

class RealTimeTriangulator:
    
    def __init__(self,camera_array:CameraArray, synchronizer:Synchronizer):
        self.camera_array = camera_array
        self.synchronizer = synchronizer
        
        self.stop_thread = Event()
        self.stop_thread.clear()
        self._sync_packet_history = []     
        self.sync_packet_in_q = Queue(-1) 
        self.synchronizer.subscribe_to_sync_packets(self.sync_packet_in_q)

        
        self.projection_matrices = Dict()
        for port, cam in self.camera_array.cameras.items():
            self.projection_matrices[port] = cam.projection_matrix


        self.thread = Thread(target=self.process_incoming, args=(), daemon=True)
        self.thread.start()
        self.running = True
    
    def process_incoming(self):
        
        while not self.stop_thread.is_set():

            sync_packet:SyncPacket = self.sync_packet_in_q.get()
            logger.info("Sync Packet Grabbed...")     

            if sync_packet is None:
                # No more sync packets after this... wind down
                self.stop_thread.set()
                logger.info("End processing of incoming sync packets...end signaled with `None` packet")

            else:    
                self._sync_packet_history.append(sync_packet)
    
                cameras, point_ids, imgs_xy = sync_packet.triangulation_inputs

                # only attempt to process if data exists
                if len(cameras) > 2:
                    # prepare for jit
                    cameras = np.array(cameras)
                    point_ids = np.array(point_ids)
                    imgs_xy = np.array(imgs_xy)
                    # only attempt to process points with multiple views
                    # iterated across the current points to find those with multiple views

                    point_id_xyz, points_xyz = triangulate_sync_index(
                        self.projection_matrices, cameras, point_ids, imgs_xy
                    )
            
                    logger.info(f"Point ID: {point_id_xyz} and xyz: {points_xyz}")
        
        self.running = False 
        
        
       
        
# helper function to avoid use of np.unique(return_counts=True) which doesn't work with jit
# @jit(nopython=True)
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


# @jit(nopython=True, parallel=True, cache=True)
def triangulate_sync_index(
    projection_matrices, current_camera_indices, current_point_id, current_img
):
    # sync_indices_xyz = List()
    point_indices_xyz = List()
    obj_xyz = List()

    unique_points, point_counts = unique_with_counts(current_point_id)
    for index in range(len(point_counts)):
        if point_counts[index] > 1:
            # triangulate that points...
            point = unique_points[index]
            points_xy = current_img[current_point_id == point]
            camera_ids = current_camera_indices[current_point_id == point]
            # logger.info(f"Calculating xyz for point {point} at sync index {sync_id}")
            # point_xyz = triangulate_simple(points_xy, camera_ids, projection_matrices)

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

            # sync_indices_xyz.append(sync_id)
            point_indices_xyz.append(point)
            obj_xyz.append(point_xyz)

    return point_indices_xyz, obj_xyz