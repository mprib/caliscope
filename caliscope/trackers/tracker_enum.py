from enum import Enum

from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.trackers.hand_tracker import HandTracker
from caliscope.trackers.holistic.holistic_tracker import HolisticTracker
from caliscope.trackers.pose_tracker import PoseTracker
from caliscope.trackers.simple_holistic_tracker import SimpleHolisticTracker

# Temporarily removed face tracker because crashing sample project
# from caliscope.trackers.face_tracker import FaceTracker


class TrackerEnum(Enum):
    HAND = HandTracker
    POSE = PoseTracker
    SIMPLE_HOLISTIC = SimpleHolisticTracker
    HOLISTIC = HolisticTracker
    CHARUCO = CharucoTracker
    # FACE = FaceTracker


if __name__ == "__main__":
    tracker_factories = [enum_member.name for enum_member in TrackerEnum]
    print(tracker_factories)
