from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, cast

from numpy.typing import NDArray


class PixelFormat(StrEnum):
    GRAY = "gray"
    BGR = "bgr"


@dataclass(frozen=True, slots=True)
class PointPacket:
    """
    This will be the primary return value of the Tracker

    Note that obj_loc will generally be `None`. These are the point positions in
    the object's frame of reference.
    It has actual values when using the Charuco tracker as these are used in the calibration.
    """

    # Using NDArray[Any] because trackers produce various dtypes (int32 for IDs, float32/64 for coords)
    object_id: NDArray[Any]  # which object (marker ID, board ID, person instance)
    keypoint_id: NDArray[Any]  # which point within the object (corner index, joint index)
    img_loc: NDArray[Any]  # x,y position of tracked point
    obj_loc: NDArray[Any] | None = None  # x,y,z in object frame of reference; primarily for intrinsic calibration
    confidence: NDArray[Any] | None = None  # may be available in some trackers..include for future

    @property
    def obj_loc_list(self) -> list[list[float | None]]:
        """
        if obj loc is none, printing data to a table format will be undermined
        this creates a list of length len(keypoint_id) that is empty so that a pd.dataframe
        can be constructed from it.
        """
        if self.obj_loc is not None:
            obj_loc_x = self.obj_loc[:, 0].tolist()
            obj_loc_y = self.obj_loc[:, 1].tolist()
            obj_loc_z = self.obj_loc[:, 2].tolist()
        else:
            length = len(self.keypoint_id) if self.keypoint_id is not None else 0
            obj_loc_x = [None] * length
            obj_loc_y = [None] * length
            obj_loc_z = [None] * length

        return cast(list[list[float | None]], [obj_loc_x, obj_loc_y, obj_loc_z])


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Raw decode output from a single camera frame."""

    cam_id: int
    frame_index: int
    frame_time: float
    frame: NDArray[Any]
    pixel_format: PixelFormat = PixelFormat.BGR


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """A decoded frame enriched with tracking results."""

    cam_id: int
    frame_index: int
    frame_time: float
    frame: NDArray[Any] | None  # None for end-of-stream markers
    points: PointPacket | None = None
    draw_instructions: Callable | None = None
    pixel_format: PixelFormat = PixelFormat.BGR

    def to_tidy_table(self, sync_index) -> dict | None:
        """
        Returns a dictionary of lists where each list is as long as the
        number of points identified on the frame;
        used for creating csv output via pandas
        """
        if self.points is not None:
            point_count = len(self.points.keypoint_id)
            if point_count > 0:
                table = {
                    "sync_index": [sync_index] * point_count,
                    "cam_id": [self.cam_id] * point_count,
                    "frame_index": [self.frame_index] * point_count,
                    "frame_time": [self.frame_time] * point_count,
                    "object_id": self.points.object_id.tolist(),
                    "keypoint_id": self.points.keypoint_id.tolist(),
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
