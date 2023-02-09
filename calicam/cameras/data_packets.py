
from dataclasses import dataclass
import numpy as np

@dataclass 
class PointPacket:
   point_id: np.ndarray=None
   img_loc: np.ndarray=None
   board_loc: np.ndarray=None 

@dataclass
class FramePacket:
   port: int
   frame_time:float
   frame: np.ndarray
   frame_index: int=None
   points: PointPacket=None 
        
@dataclass
class SyncPacket:
   sync_index: int
   frame_packets: dict[FramePacket]