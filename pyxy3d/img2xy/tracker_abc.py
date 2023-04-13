
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class PointPacket:
    point_id: np.ndarray = None
    img_loc: np.ndarray = None
    obj_loc: np.ndarray = None # x,y,z in object frame of reference; primarily for calibration
    confidence: np.ndarray = None
    # board_loc: np.ndarray = None
    
    
class Tracker(ABC):
    
    @abstractmethod
    def process_frame(self, frame:np.ndarray)->PointPacket:
        pass

    @abstractmethod
    def get_point_names(self)->dict:
        pass
        