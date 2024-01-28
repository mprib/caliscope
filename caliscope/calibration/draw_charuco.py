# a set of helper functions meant to provide visual feedback
# regarding the capture history and corner identification of
# the charuco board
import caliscope.logger
from itertools import combinations
import cv2
logger = caliscope.logger.get(__name__)

def grid_history(frame, ids, img_locs, connected_corners):
    """
    add the history of captured boards so that the user can see which ares of the camera FOV may not have data
    """
    
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

        cv2.line(frame, point_1, point_2, (255, 165, 0), 3)

    return frame


