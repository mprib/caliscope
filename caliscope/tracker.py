from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
from caliscope.packets import PointPacket, XYZPacket
from pyqtgraph.opengl import GLLinePlotItem
import pyqtgraph as pg


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
    def scatter_draw_instructions(self, point_id: int) -> dict:
        """
        Maps point_id to a dictionary of parameters used to draw circles on frames for visual feedback.

        As an example, the dictionary could have the form: {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        The parameters `radius`, `color`, and `thickness` are used downstream in a call to `cv2.circle`
        to place the tracked point on the frame. See `FramePacket` below.
        """
        pass

    def get_connected_points(self) -> set[tuple[int, int]]:
        """
        OPTIONAL METHOD
        used for 2d drawing purposes elsewhere. Specify which
        points (if any) should have a line connecting them
        {(point_id_A, point_id_B),etc...}

        currently only implemented for charuco...
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


@dataclass(slots=True, frozen=True)
class Segment:
    name: str
    color: str  # one of: r, g, b, c, m, y, k, w
    point_A: str  # name of landmark
    point_B: str  # name of landmark
    width: float = 1  # note that this does not scale with zoom level... should probably just stick with 1


@dataclass(slots=False, frozen=False)
class WireFrameView:
    segments: [Segment]
    point_names: dict[str:int]  # map landmark name to landmark id

    def __post_init__(self):
        self.line_plots = {}
        for segment in self.segments:
            self.line_plots[segment.name] = GLLinePlotItem(
                color=pg.mkColor(segment.color), width=segment.width, mode="lines"
            )

        self.point_ids = {value:key for key,value in self.point_names.items()}
    def set_points(self, xyz_packet: XYZPacket):
        for segment in self.segments:
            A_id = self.point_ids[segment.point_A]
            B_id = self.point_ids[segment.point_B]
            self.line_plots[segment.name].setData(
                pos=xyz_packet.get_segment_ends(A_id,B_id))
