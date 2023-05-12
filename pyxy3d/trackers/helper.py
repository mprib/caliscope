import cv2
import numpy as np

def apply_rotation(frame: np.ndarray, rotation_count: int) -> np.ndarray:
    if rotation_count == 0:
        pass
    elif rotation_count in [1, -3]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_count in [2, -2]:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_count in [-1, 3]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame


def unrotate_points(
    xy: np.ndarray, rotation_count: int, frame_width: int, frame_height: int
) -> np.ndarray:
    xy_unrotated = xy.copy()

    if rotation_count == 0 or len(xy) ==0:
        pass
    elif rotation_count in [1, -3]:
        # Reverse of 90 degrees clockwise rotation
        xy_unrotated[:, 0], xy_unrotated[:, 1] = xy[:, 1], frame_width - xy[:, 0]

    elif rotation_count in [2, -2]:
        # NOTE: have not verified this with a test case
        # Reverse of 180 degrees rotation
        xy_unrotated[:, 0], xy_unrotated[:, 1] = (
            frame_width - xy[:, 0],
            frame_height - xy[:, 1],
        )
    elif rotation_count in [-1, 3]:
        # Reverse of 90 degrees counter-clockwise rotation
        xy_unrotated[:, 0], xy_unrotated[:, 1] = frame_height - xy[:, 1], xy[:, 0]

    return xy_unrotated
