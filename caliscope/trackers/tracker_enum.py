from enum import Enum


class TrackerEnum(str, Enum):
    """
    Enum of available tracker implementations.
    Used for selecting trackers in GUI and CLI.
    """

    CHARUCO = "CHARUCO"
    HAND = "HAND"
    POSE = "POSE"
    HOLISTIC = "HOLISTIC"
    SIMPLE_HOLISTIC = "SIMPLE_HOLISTIC"
    ARUCO = "ARUCO"


if __name__ == "__main__":
    tracker_factories = [enum_member.name for enum_member in TrackerEnum]
    print(tracker_factories)
