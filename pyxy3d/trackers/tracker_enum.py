from enum import Enum


from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.trackers.hand_tracker  import  HandTracker
from pyxy3d.trackers.pose_tracker import  PoseTracker
from pyxy3d.trackers.holistic_tracker import HolisticTracker

class TrackerEnum(Enum):
    HAND = HandTracker
    POSE = PoseTracker
    HOLISTIC = HolisticTracker
    CHARUCO = CharucoTracker

    
if __name__ == "__main__":
    tracker_factories = [enum_member.name for enum_member in TrackerEnum]
    print(tracker_factories)
