from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
from caliscope.packets import XYZPacket
import numpy as np
from caliscope.trackers.tracker_enum import TrackerEnum

@dataclass
class MotionTrial:
    """
    Motion trial loaded in from output csv
    """

    xyz_csv: Path    
    xyz_packets:  dict = field(default_factory=dict[int:XYZPacket])
    
    def __post_init__(self):
        # assert(isinstance(self.xyz_csv,Path))
        # assert(self.xyz_csv.exists())

        tracker_name = self.xyz_csv.stem[4:]  # peel off "xyz_"
        self.tracker = TrackerEnum[tracker_name].value()

        if hasattr(self.tracker, "wireframe"):
            self.wireframe = self.tracker.wireframe
        else:
            self.wireframe = None

        self.xyz_df = pd.read_csv(self.xyz_csv, engine="pyarrow")
        sync_indices = self.xyz_df["sync_index"].unique()

        self.start_index = sync_indices.min()
        self.end_index = sync_indices.max()
        self.xyz_packets = {}
                
        if len(sync_indices) ==0:
            self.is_empty = True
        else:
            self.is_empty = False

    def get_xyz(self,sync_index:int)->XYZPacket:
        """
        Cache packets as they are initially read off
        """
        if sync_index not in self.xyz_packets:
            current_sync_index = self.xyz_df["sync_index"] == sync_index
            point_ids = self.xyz_df["point_id"][current_sync_index]
                
            x = self.xyz_df["x_coord"][current_sync_index]
            y = self.xyz_df["y_coord"][current_sync_index]               
            z = self.xyz_df["z_coord"][current_sync_index]

            xyz = np.column_stack([x,y,z])
            self.xyz_packets[sync_index] = XYZPacket(sync_index=sync_index, point_ids=point_ids,point_xyz=xyz)

        return self.xyz_packets[sync_index]
    
    def update_wireframe(self, sync_index:int):
        xyz_packet = self.get_xyz(sync_index)
        self.wireframe.set_points(xyz_packet)