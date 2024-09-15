from enum import Enum

from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.trackers.face_tracker import FaceTracker
from caliscope.trackers.hand_tracker import HandTracker
from caliscope.trackers.holistic.holistic_tracker import HolisticTracker
from caliscope.trackers.holistic_opensim_tracker import HolisticOpenSimTracker
from caliscope.trackers.pose_tracker import PoseTracker


class TrackerEnum(Enum):
    HAND = HandTracker
    POSE = PoseTracker
    HOLISTIC_OPENSIM = HolisticOpenSimTracker
    HOLISTIC = HolisticTracker
    CHARUCO = CharucoTracker
    FACE = FaceTracker


if __name__ == "__main__":
    tracker_factories = [enum_member.name for enum_member in TrackerEnum]
    print(tracker_factories)
