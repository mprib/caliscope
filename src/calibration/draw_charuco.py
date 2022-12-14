# a set of helper functions meant to provide visual feedback
# regarding the capture history and corner identification of
# the charuco board
from itertools import combinations

import cv2
import numpy as np


def grid_history(frame, ids, img_locs, connected_corners):
    ids = ids[:,0]
    img_locs = img_locs[:,0]
    
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


def corners(frame, locs):
    # if type(locs) == list:
    #     locs = np.array(locs, dtype=np.float32)

    if locs.any():
        for coord in locs[:,0]:
            # coord = list(coord)
            # print(frame.shape[1])
            x = round(coord[0])
            y = round(coord[1])

            cv2.circle(frame, (x, y), 5, (0, 0, 220), 3)
            # cv2.putText(self.frame,str(ID), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5,(220,0,0), 3)
    return frame
