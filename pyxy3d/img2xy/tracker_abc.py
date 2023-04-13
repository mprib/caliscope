
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from pyxy3d.cameras.data_packets import PointPacket
    
class Tracker(ABC):
    
    @abstractmethod
    def process_frame(self, frame:np.ndarray)->PointPacket:
        pass

    @abstractmethod
    def get_point_names(self)->dict:
        pass
        