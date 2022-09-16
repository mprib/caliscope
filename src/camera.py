
# %%

import cv2 as cv
import time 
import numpy as np
import time

from collections import defaultdict
from itertools import combinations

import charuco

class Camera():
    """
    Allows collection of calibration data for an individual camera
    """

    def __init__(self, input_stream,stream_name):
        """
        Initialize input stream for individual video calibration
        """

        self.input_stream = input_stream
        self.stream_name = stream_name
        self.image_size = []
        self.calibration_corners = []
        self.calibration_ids = []
        self.objective_corners = []



    def collect_calibration_corners(self, board_threshold, charuco, charuco_inverted=False, time_between_cal=1):
        """
        Charuco: a cv2 charuco board
        board_threshold: percent of board corners that must be represented to record
        """

        # store charuco used for calibration
        self.charuco = charuco

        capture = cv.VideoCapture(self.input_stream)
        # capture = cv.VideoCapture(self.input_stream)

        min_points_to_process = int(len(self.charuco.board.chessboardCorners) * board_threshold)
        connected_corners = self.charuco.get_connected_corners()

        # open the capture stream 
        while True:
        
            read_success, frame = capture.read()

            # initialize parameters on first loop
            if len(self.image_size) == 0 :
                self.image_size = frame.shape
                self.grid_capture_history =  np.zeros(self.image_size, dtype='uint8')
                last_calibration_time = time.time()

            # check for charuco corners in the image
            found_corner, charuco_corners, charuco_corner_ids = self.find_corners(
                frame, 
                self.charuco.board, 
                charuco_inverted)

            # if charuco corners are detected
            if found_corner:

                # draw them on the frame to visualize
                frame = cv.aruco.drawDetectedCornersCharuco(
                    image = frame,
                    charucoCorners=charuco_corners,
                    cornerColor = (120,255,0))
                
                # add to the calibration corners if enough corners observed
                # and enough time  has passed from the last "snapshot"
                enough_corners = len(charuco_corner_ids) > min_points_to_process
                enough_time_from_last_cal = time.time() > last_calibration_time+time_between_cal

                if enough_corners and enough_time_from_last_cal:

                    # store the corners and IDs
                    self.calibration_corners.append(charuco_corners)
                    self.calibration_ids.append(charuco_corner_ids)

                    # objective corner position in a board frame of reference
                    board_FOR_corners = self.charuco.board.chessboardCorners[charuco_corner_ids, :]
                    self.objective_corners.append(board_FOR_corners)

                    # 
                    self.draw_charuco_outline(charuco_corners, charuco_corner_ids, connected_corners)

                    last_calibration_time = time.time()

            # merge calibration footprint and live frame
            alpha = 1
            beta = 1
            merged_frame = cv.addWeighted(frame, alpha, self.grid_capture_history, beta, 0)
            cv.imshow(self.stream_name, merged_frame)


            # end capture when enough grids collected
            if cv.waitKey(5) == 27: # ESC to stop 
                capture.release()
                cv.destroyWindow(self.stream_name)
                break
    
    def find_corners(self, frame, charuco, charuco_inverted):
        
        # invert the frame for detection if needed
        if charuco_inverted:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert to gray
            frame = ~frame  # invert
        
        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv.aruco.detectMarkers(
            frame, 
            charuco.dictionary)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 3:
            success, charuco_corners, charuco_corner_ids = cv.aruco.interpolateCornersCharuco(
                aruco_corners,
                aruco_ids,
                frame,
                charuco)
            
            if charuco_corners is not None:
                return True, charuco_corners, charuco_corner_ids
            else:
                return False, None, None

        else:
            return False, None, None


    def draw_charuco_outline(self, charuco_corners, charuco_ids, connected_corners):
        """
        Given a frame and the location of the charuco board corners within in,
        draw a line connecting the outer bounds of the detected corners
        """

        possible_pairs = {pair for pair in combinations(charuco_ids.squeeze().tolist(),2)}
        connected_pairs = connected_corners.intersection(possible_pairs)

        # build dictionary of corner positions:
        observed_corners = {}
        for id, crnr in zip(charuco_ids.squeeze(), charuco_corners.squeeze()):
            observed_corners[id] = (round(crnr[0]), round(crnr[1]))
        
        # print(corners)

        # drawn_boards = len(self.calibration_ids[stream_name])

        for pair in connected_pairs:
            point_1 = observed_corners[pair[0]]
            point_2 = observed_corners[pair[1]]

            cv.line(self.grid_capture_history,point_1, point_2, (255, 165, 0), 1)
   

    def calibrate(self):
        objpoints = 

        cv.calibrateCamera()
            


    