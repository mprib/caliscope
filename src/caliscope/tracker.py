from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from pyqtgraph.opengl import GLLinePlotItem

from dataclasses import dataclass
from caliscope.packets import PointPacket, XYZPacket


class Tracker(ABC):
    @property
    def name(self) -> str:
        """
        returns the tracker name
        This name should align with the label used by TrackerEnum
        Used for file naming creation
        """
        return "Name Me"

    @abstractmethod
    def get_points(self, frame: np.ndarray, port: int = 0, rotation_count: int = 0) -> PointPacket:
        """
        frame: np.ndarray from reading an OpenCV capture object

        port: integer value usd to track which camera the frame originates from
              Default 0 for trackers that don't need camera identification

        rotation count: used to indicate the orientation of the image (e.g. rotateed 90 degrees left or right)
                        Some tracking algorithms expect images to be "upright", so this can be used to align the image
                        Default 0 for trackers that are rotation invariant (e.g. ArUco)

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

    @property
    def wireframe_toml_path(self) -> Path | None:
        """
        OPTIONAL: Path to wireframe definition TOML, or None if no wireframe.
        This is a UI concern - the tracker just provides the path.
        """
        return None

    def get_connected_points(self) -> set[tuple[int, int]]:
        """
        OPTIONAL METHOD
        used for 2d drawing purposes elsewhere. Specify which
        points (if any) should have a line connecting them
        {(point_id_A, point_id_B),etc...}

        currently only implemented for charuco...
        """
        return set()


@dataclass(slots=True, frozen=True)
class Segment:
    name: str
    color: str  # one of: r, g, b, c, m, y, k, w
    point_A: str  # name of landmark
    point_B: str  # name of landmark
    width: float = 1  # note that this does not scale with zoom level... should probably just stick with 1


@dataclass(slots=False, frozen=False)
class WireFrameView:
    segments: list[Segment]
    point_names: dict[str, int]  # map landmark name to landmark id

    def __post_init__(self):
        self.line_plots: dict[str, GLLinePlotItem] = {}
        for segment in self.segments:
            self.line_plots[segment.name] = GLLinePlotItem(
                color=pg.mkColor(segment.color), width=segment.width, mode="lines"
            )

        # Same as point_names: point_name (str) -> point_id (int)
        # Used in set_points() to look up IDs by landmark name
        self.point_ids: dict[str, int] = dict(self.point_names)

    def set_points(self, xyz_packet: XYZPacket):
        for segment in self.segments:
            A_id = self.point_ids[segment.point_A]
            B_id = self.point_ids[segment.point_B]
            self.line_plots[segment.name].setData(pos=xyz_packet.get_segment_ends(A_id, B_id))
