from dataclasses import dataclass
import numpy as np


@dataclass
class PointPacket:
    """
    This will be the primary return value of the Tracker Protocol
    A calleable that receives an image frame and returns a point_packet
    """
    point_id: np.ndarray = None
    img_loc: np.ndarray = None
    board_loc: np.ndarray = None


@dataclass
class FramePacket:
    """
    Holds the data for a single frame from a camera, including the frame itself,
    the frame time and the points if they were generated
    """

    port: int
    frame_time: float
    frame: np.ndarray
    frame_index: int = None
    points: PointPacket = None

    def to_tidy_table(self, sync_index):
        """
        Returns a dictionary of lists where each list is as long as the 
        number of points identified on the frame
        """
        
        point_count = len(self.points.point_id)
        if point_count > 0:
            table = {
                    "sync_index": [sync_index]*point_count,
                    "port":[self.port]*point_count,
                    "frame_index":[self.frame_index]*point_count,
                    "frame_time":[self.frame_time]*point_count,
                    "point_id":self.points.point_id.tolist(),
                    "img_loc_x":self.points.img_loc[:,0].tolist(),
                    "img_loc_y": self.points.img_loc[:,1].tolist(),
                    "board_loc_x":self.points.board_loc[:,0].tolist(),
                    "board_loc_y":self.points.board_loc[:,1].tolist()
                    }       
        else:
            table = None 
        return table
@dataclass
class SyncPacket:
    """
    SyncPacket holds syncronized frame packets.
    """

    sync_index: int
    frame_packets: dict


    @property
    def triangulation_inputs(self):
        """
        returns three key items used by the triangulation functions 
            cameras: a list of the camera ids associated with each reported 2d point
            point_ids: the point id associated with each 2d ponit
            img_xy: the 2d image points themselves
        
        """
        cameras = []
        point_ids = []
        img_xy = []

        for port, packet in self.frame_packets.items():
            cameras.extend([port]*len(packet.points.point_id))
            point_ids.extend(packet.points.point_id.tolist())            
            img_xy.extend(packet.points.img_loc.tolist())
        
        return cameras, point_ids,img_xy