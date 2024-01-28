from dataclasses import dataclass
import numpy as np
from numba.typed import List
from abc import ABC, abstractmethod
import cv2


@dataclass(frozen=True, slots=True)
class PointPacket:
    """
    This will be the primary return value of the Tracker

    Note that obj_loc will generally be `None`. These are the point positions in
    the object's frame of reference.
    It has actual values when using the Charuco tracker as these are used in the calibration.
    """

    point_id: np.ndarray = (
        None  # unique point id that aligns with Tracker.get_point_names()
    )
    img_loc: np.ndarray = None  # x,y position of tracked point
    obj_loc: np.ndarray = (
        None  # x,y,z in object frame of reference; primarily for calibration
    )
    confidence: np.ndarray = None  # may be available in some trackers..include for potentnial downstream calculations

    @property
    def obj_loc_list(self) -> List[List]:
        """
        if obj loc is none, printing data to a table format will be undermined
        this creates a list of length len(point_id) that is empty so that a pd.dataframe
        can be constructed from it.
        """
        if self.obj_loc is not None:
            obj_loc_x = self.obj_loc[:, 0].tolist()
            obj_loc_y = self.obj_loc[:, 1].tolist()
        else:
            length = len(self.point_id)
            obj_loc_x = [None] * length
            obj_loc_y = [None] * length

        return [obj_loc_x, obj_loc_y]


class Tracker(ABC):
    @property
    def name(self) -> str:
        """
        returns the tracker name
        This name should align with the label used by TrackerEnum
        Used for file naming creation
        """
        pass

    @abstractmethod
    def get_points(
        self, frame: np.ndarray, port: int, rotation_count: int
    ) -> PointPacket:
        """
        frame: np.ndarray from reading an OpenCV capture object

        port: integer value usd to track which camera the frame originates from

        rotation count: used to indicate the orientation of the image (e.g. rotateed 90 degrees left or right)
                        Some tracking algorithms expect images to be "upright", so this can be used to align the image

                        The function `apply_rotation` from `caliscope.trackers.helper` can correctly orient the image
                        The function `unrotate_points` from the same module can convert any tracked points back into
                        the original orientation
        """
        pass

    @abstractmethod
    def get_point_name(self, point_id: int) -> str:
        """
        Maps point_id to a name
        Used for saving out data with sensible headers.
        """
        pass

    @abstractmethod
    def get_point_id(self, point_name: str) -> int:
        """
        Maps point name to point_id
        """
        pass

    @abstractmethod
    def scatter_draw_instructions(self, point_id: int) -> dict:
        """
        Maps point_id to a dictionary of parameters used to draw circles on frames for visual feedback.

        As an example, the dictionary could have the form: {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        The parameters `radius`, `color`, and `thickness` are used downstream in a call to `cv2.circle`
        to place the tracked point on the frame. See `FramePacket` below.
        """
        pass

    def get_connected_points(self) -> dict[str : tuple[int, int, tuple[int, int]]]:
        """
        OPTIONAL METHOD
        used for drawing purposes elsewhere. Specify which
        points (if any) should have a line connecting them
        {SegmentName:(pointNameA, pointNameB, )}
        """
        pass

    @property
    def metarig_mapped(self):
        """
        OPTIONAL PROPERTY

        Defaults to false and can be overriden to True
        Used to ensure that metarig_config creation is not presented as
        an option in GUI
        """
        return False

    @property
    def metarig_symmetrical_measures(self):
        """
        OPTIONAL PROPERTY

        a dictionary of key: value in the form Measure_Name:[pointA, pointB]
        when processed, the mean distances (excluding outliers) of both
        left_pointA,left_pointB and right_pointA, right_pointB will be calculated.
        The mean of the two sides will be taken
        """
        raise NotImplementedError(
            f"Tracker {self.name} has not provided its measures for configuring a metarig"
        )

    @property
    def metarig_bilateral_measures(self):
        """
        OPTIONAL PROPERTY

        a dictionary of key: value in the form Measure_Name:[side_pointA, side_pointB]
        when processed, mean distance (excluding outliers) between the two points will be calculated and stored as the measure
        """
        raise NotImplementedError(
            f"Tracker {self.name} has not provided its measures for configuring a metarig"
        )


@dataclass(frozen=True, slots=True)
class FramePacket:
    """
    Holds the data for a single frame from a camera, including the frame itself,
    the frame time and the points if they were generated
    """

    port: int
    frame_index: int
    frame_time: float
    frame: np.ndarray
    points: PointPacket = None
    draw_instructions: callable = None

    def to_tidy_table(self, sync_index) -> dict:
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
                }
            else:
                table = None
        else:
            table = None
        return table

    @property
    def frame_with_points(self):
        if self.points is not None:
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
    frame_packets: dict

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


@dataclass(slots=True,frozen=True)
class XYZPacket:
    sync_index: int
    point_ids: np.ndarray  # (n,1)
    point_xyz: np.ndarray  # (n,3)
        
    def get_point_xyz(self, point_id:int)->np.ndarray:
        return self.point_xyz[self.point_ids==point_id]

    def get_segment_ends(self, point_id_A:int, point_id_B:int)->np.ndarray:
        return np.vstack([self.get_point_xyz(point_id_A), self.get_point_xyz(point_id_B)])
    