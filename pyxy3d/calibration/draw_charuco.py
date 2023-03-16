# a set of helper functions meant to provide visual feedback
# regarding the capture history and corner identification of
# the charuco board
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from itertools import combinations

import cv2

def grid_history(frame, ids, img_locs, connected_corners):
    """
    NOTE: This is used in the monocalibrator, which is somewhat deliberately broken
    right now while I sort out an alternate calibration framework. The long
    term plan is to re-integrate the monocalibration workflow as a final "touch-up"
    step in the event that the intrinsics have a poor RMSE of reprojection
    """
    # ids = ids[:,0] pretty sure no longer needed
    # img_locs = img_locs[:,0] pretty sure no longer needed
    
    possible_pairs = {pair for pair in combinations(ids, 2)}
    connected_pairs = connected_corners.intersection(possible_pairs)

    # build dictionary of corner positions:
    observed_corners = {}
    for crnr_id, crnr in zip(ids, img_locs):
        observed_corners[crnr_id] = (round(crnr[0]), round(crnr[1]))

    # add them to the visual representation of the grid capture history
    for pair in connected_pairs:
        point_1 = observed_corners[pair[0]]
        point_2 = observed_corners[pair[1]]

        cv2.line(frame, point_1, point_2, (255, 165, 0), 1)

    return frame


def corners(frame_packet):
    frame = frame_packet.frame
    if frame_packet.points is not None:
        locs = frame_packet.points.img_loc
        for coord in locs:
            x = round(coord[0])
            y = round(coord[1])

            cv2.circle(frame, (x, y), 5, (0, 0, 220), 3)
