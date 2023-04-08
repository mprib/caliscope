

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

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
                
            self._sync_packet_history.append(sync_packet)
        
        self.running = False        
        
