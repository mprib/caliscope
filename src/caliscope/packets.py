from dataclasses import dataclass
from typing import Any, Callable, cast

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PointPacket:
    """
    This will be the primary return value of the Tracker

    Note that obj_loc will generally be `None`. These are the point positions in
    the object's frame of reference.
    It has actual values when using the Charuco tracker as these are used in the calibration.
    """

    # Using NDArray[Any] because trackers produce various dtypes (int32 for IDs, float32/64 for coords)
    point_id: NDArray[Any]  # unique point id that aligns with Tracker.get_point_names()
    img_loc: NDArray[Any]  # x,y position of tracked point
    obj_loc: NDArray[Any] | None = None  # x,y,z in object frame of reference; primarily for intrinsic calibration
    confidence: NDArray[Any] | None = None  # may be available in some trackers..include for future

    @property
    def obj_loc_list(self) -> list[list[float | None]]:
        """
        if obj loc is none, printing data to a table format will be undermined
        this creates a list of length len(point_id) that is empty so that a pd.dataframe
        can be constructed from it.
        """
        if self.obj_loc is not None:
            obj_loc_x = self.obj_loc[:, 0].tolist()
            obj_loc_y = self.obj_loc[:, 1].tolist()
            obj_loc_z = self.obj_loc[:, 2].tolist()
        else:
            length = len(self.point_id) if self.point_id is not None else 0
            obj_loc_x = [None] * length
            obj_loc_y = [None] * length
            obj_loc_z = [None] * length

        return cast(list[list[float | None]], [obj_loc_x, obj_loc_y, obj_loc_z])


@dataclass(frozen=True, slots=True)
class FramePacket:
    """
    Holds the data for a single frame from a camera, including the frame itself,
    the frame time and the points if they were generated
    """

    port: int
    frame_index: int
    frame_time: float
    frame: NDArray[np.float64]
    points: PointPacket | None = None
    draw_instructions: Callable | None = None

    def to_tidy_table(self, sync_index) -> dict | None:
        """
        Returns a dictionary of lists where each list is as long as the
        number of points identified on the frame;
        used for creating csv output via pandas
        """
        if self.points is not None:
            point_count = len(self.points.point_id)
            if point_count > 0:
                table = {
                    "sync_index": [sync_index] * point_count,
                    "port": [self.port] * point_count,
                    "frame_index": [self.frame_index] * point_count,
                    "frame_time": [self.frame_time] * point_count,
                    "point_id": self.points.point_id.tolist(),
                    "img_loc_x": self.points.img_loc[:, 0].tolist(),
                    "img_loc_y": self.points.img_loc[:, 1].tolist(),
                    "obj_loc_x": self.points.obj_loc_list[0],
                    "obj_loc_y": self.points.obj_loc_list[1],
                    "obj_loc_z": self.points.obj_loc_list[2],
                }
            else:
                table = None
        else:
            table = None
        return table

    @property
    def frame_with_points(self):
        if self.points is not None and self.draw_instructions is not None:
            drawn_frame = self.frame.copy()
            ids = self.points.point_id
            locs = self.points.img_loc
            for _id, coord in zip(ids, locs):
                x = round(coord[0])
                y = round(coord[1])

                # draw instructions are a method of Tracker object
                params = self.draw_instructions(_id)
                cv2.circle(
                    drawn_frame,
                    (x, y),
                    params["radius"],
                    params["color"],
                    params["thickness"],
                )
        else:
            drawn_frame = self.frame

        return drawn_frame


@dataclass(frozen=True, slots=True)
class SyncPacket:
    """
    SyncPacket holds syncronized frame packets.
    """

    sync_index: int
    frame_packets: dict[int, FramePacket | None]  # port: frame_packet

    @property
    def triangulation_inputs(self):
        """
        returns three key items used by the triangulation functions
            cameras: a list of the camera ids associated with each reported 2d point
            point_ids: the point id associated with each 2d point
            img_xy: the 2d image points themselves

        """
        cameras = []
        point_ids = []
        img_xy = []

        for port, packet in self.frame_packets.items():
            if packet is not None and packet.points is not None:
                cameras.extend([port] * len(packet.points.point_id))
                point_ids.extend(packet.points.point_id.tolist())
                img_xy.extend(packet.points.img_loc.tolist())

        return cameras, point_ids, img_xy

    @property
    def dropped(self):
        """
        convencience method to ease tracking of dropped frame rate within the synchronizer
        """
        temp_dict = {}
        for port, packet in self.frame_packets.items():
            if packet is None:
                temp_dict[port] = 1
            else:
                temp_dict[port] = 0
        return temp_dict

    @property
    def frame_packet_count(self):
        count = 0
        for port, packet in self.frame_packets.items():
            if packet is not None:
                count += 1
        return count


@dataclass(slots=True, frozen=True)
class XYZPacket:
    sync_index: int
    point_ids: NDArray[np.float64]  # (n,1)
    point_xyz: NDArray[np.float64]  # (n,3)

    def get_point_xyz(self, point_id: int) -> np.ndarray:
        return self.point_xyz[self.point_ids == point_id]

    def get_segment_ends(self, point_id_A: int, point_id_B: int) -> np.ndarray:
        return np.vstack([self.get_point_xyz(point_id_A), self.get_point_xyz(point_id_B)])
