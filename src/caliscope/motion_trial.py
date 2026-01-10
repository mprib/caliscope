import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from caliscope.packets import XYZPacket
from caliscope.tracker import Tracker, WireFrameView
from caliscope.trackers.tracker_enum import TrackerEnum

logger = logging.getLogger(__name__)


def _can_instantiate_without_args(cls: type) -> bool:
    """Check if a class can be instantiated with no arguments.

    Inspects __init__ signature and returns False if any parameter
    (other than 'self') lacks a default value.
    """
    sig = inspect.signature(cls.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            return False
    return True


@dataclass
class MotionTrial:
    """
    Motion trial loaded in from output csv
    """

    xyz_csv: Path
    xyz_packets: dict[int, XYZPacket] = field(default_factory=dict)

    def __post_init__(self):
        # assert(isinstance(self.xyz_csv,Path))
        # assert(self.xyz_csv.exists())

        tracker_name = self.xyz_csv.stem[4:]  # peel off "xyz_"
        self.tracker: Tracker | None = None
        self.wireframe: WireFrameView | None = None
        try:
            # We want the tracker's wireframe for 3D visualization, but:
            # - Only HolisticTracker defines a wireframe attribute
            # - Some trackers (e.g., CharucoTracker) require constructor arguments
            tracker_cls = TrackerEnum[tracker_name].value

            if _can_instantiate_without_args(tracker_cls):
                # Runtime check above guarantees no required args; type checker can't follow this
                self.tracker = tracker_cls()  # type: ignore[call-arg]
                # Returns tracker.wireframe if it exists, otherwise None
                self.wireframe = getattr(self.tracker, "wireframe", None)
        except Exception as e:
            logger.exception(f"{e}: Using tracker of type {tracker_name}, but unable to create")

        self.xyz_df = pd.read_csv(self.xyz_csv, engine="pyarrow")
        sync_indices = self.xyz_df["sync_index"].unique()

        self.start_index = sync_indices.min()
        self.end_index = sync_indices.max()
        self.xyz_packets = {}

        if len(sync_indices) == 0:
            self.is_empty = True
        else:
            self.is_empty = False

    def get_xyz(self, sync_index: int) -> XYZPacket:
        """
        Cache packets as they are initially read off
        """
        if sync_index not in self.xyz_packets:
            current_sync_index = self.xyz_df["sync_index"] == sync_index
            point_ids = self.xyz_df["point_id"][current_sync_index].to_numpy(dtype=np.float64)

            x = self.xyz_df["x_coord"][current_sync_index]
            y = self.xyz_df["y_coord"][current_sync_index]
            z = self.xyz_df["z_coord"][current_sync_index]

            xyz = np.column_stack([x, y, z])
            self.xyz_packets[sync_index] = XYZPacket(sync_index=sync_index, point_ids=point_ids, point_xyz=xyz)

        return self.xyz_packets[sync_index]

    def update_wireframe(self, sync_index: int):
        xyz_packet = self.get_xyz(sync_index)
        if self.wireframe is not None:
            self.wireframe.set_points(xyz_packet)
