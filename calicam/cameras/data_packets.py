from dataclasses import dataclass
import numpy as np


@dataclass
class PointPacket:
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


@dataclass
class SyncPacket:
    """
    SyncPacket holds syncronized frame packets.
    """

    sync_index: int
    frame_packets: dict

    def get_overlap(self, pair):
        """
        For use with the omniframe. Provide a simple interface to retrieve, for a
        given pair of ports, the ids that are common between them and the locations
        of the images.

        This is primarily of use for the stereocalibration process
        """
        portA = pair[0]
        portB = pair[1]

        common_ids = self.get_common_ids(portA, portB)

    def get_common_ids(self, portA, portB):

        if self.frame_packets[portA] and self.frame_packets[portB]:
            ids_A = self.frame_packets[portA].points.point_id
            ids_B = self.frame_packets[portB].points.point_id
            common_ids = np.intersect1d(ids_A, ids_B)
            common_ids = common_ids.tolist()

        else:
            common_ids = []

    def get_common_locs(self, port, common_ids):
        """Pull out objective location and image location of board corners for
        a port that are on the list of common ids"""

        ids = self.frame_packets[port].points.point_id.tolist()
        img_loc = self.frame_packets[port].points.img_loc.tolist()
        board_loc = self.frame_packets[port].points.board_loc.tolist()

        common_img_loc = []
        common_board_loc = []

        for crnr_id, img, obj in zip(ids, img_loc, board_loc):
            if crnr_id in common_ids:
                common_board_loc.append(img)
                common_img_loc.append(obj)

        return common_img_loc, common_board_loc


@dataclass
class PairedPointsPacket:
    """The points shared by two FramePointsPackets"""

    sync_index: int

    port_A: int
    port_B: int

    time_A: float
    time_B: float

    point_id: np.ndarray

    loc_board_x: np.ndarray
    loc_board_y: np.ndarray

    loc_img_x_A: np.ndarray
    loc_img_y_A: np.ndarray

    loc_img_x_B: np.ndarray
    loc_img_y_B: np.ndarray

    @property
    def pair(self):
        return (self.port_A, self.port_B)

    def to_dict(self):
        """A method that can be used by the caller of the point stream to create a
        dictionary of lists that is well-formatted to be turned into a tidy dataframe
        for export to csv."""

        packet_dict = {}
        length = len(self.point_id)
        packet_dict["sync_index"] = [self.sync_index] * length
        packet_dict["port_A"] = [self.port_A] * length
        packet_dict["port_B"] = [self.port_B] * length
        packet_dict["time_A"] = [self.time_A] * length
        packet_dict["time_B"] = [self.time_B] * length
        packet_dict["point_id"] = self.point_id.tolist()
        packet_dict["loc_board_x"] = self.loc_board_x.tolist()
        packet_dict["loc_board_y"] = self.loc_board_y.tolist()
        packet_dict["loc_img_x_A"] = self.loc_img_x_A.tolist()
        packet_dict["loc_img_y_A"] = self.loc_img_y_A.tolist()
        packet_dict["loc_img_x_B"] = self.loc_img_x_B.tolist()
        packet_dict["loc_img_y_B"] = self.loc_img_y_B.tolist()

        return packet_dict
