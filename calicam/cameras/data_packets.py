
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
    points: PointPacket=None 
        
    