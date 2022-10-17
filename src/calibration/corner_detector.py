# The purpose of this module will be to construct a worker for the real time 
# device that can look at a picure and identify/collect the corners. 
# There may be a mixed functionality here...I'm not sure. Between the corner
# detector and the corner drawer...like, there will need to be something that
# accumulates a frame of corners to be drawn onto the displayed frame.

import cv2
import time
import numpy as np
from itertools import combinations

from charuco import Charuco

class IntrinsicCalibrator:

    def __init__(self, charuco, image_size, board_threshold=.8):
    
        self.charuco = charuco
        self.image_size = image_size

        self.corner_loc_img = []
        self.corner_loc_obj = []
        self.corner_ids = []

        self.min_points_to_process = int(len(self.charuco.board.chessboardCorners) * board_threshold)
        self.connected_corners = self.charuco.get_connected_corners()
        self.last_calibration_time = time.time()    # need to initialize to *something*
        # for subpixel corner correction 
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self._conv_size = (11, 11) # Don't make this too large.


        self._grid_capture_history =  np.zeros(image_size, dtype='uint8')



    def find_corners(self, frame): 

        # invert the frame for detection if needed
        if self.charuco.inverted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
            frame = ~frame  # invert
        
        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(
            frame, 
            self.charuco.board.dictionary)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 3:
            success, charuco_corners, charuco_corner_ids = cv2.aruco.interpolateCornersCharuco(
                aruco_corners,
                aruco_ids,
                frame,
                self.charuco.board)
            
            if charuco_corners is not None:
                return True, charuco_corners, charuco_corner_ids
            else:
                return False, None, None

        else:
            return False, None, None

    def collect_corners(self, frame, wait_time=1):
        #This function does too many things. Way too many things.
        # 
        # # check for charuco corners in the image
        crnr_found, corners, ids = detector.find_corners(frame)

        # invert the frame for detection if needed
        if self.charuco.inverted:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
            gray = ~gray  # invert

        # Draw Corners if detected
        if crnr_found:
            # draw them on the frame to visualize
            frame = cv2.aruco.drawDetectedCornersCharuco(
                                image = frame,
                                charucoCorners=corners,
                                cornerColor = (120,255,0))

            # add to the calibration corners if enough corners observed
            # and enough time  has passed from the last "snapshot"
            enough_corners = len(ids) > self.min_points_to_process
            enough_time_from_last_cal = time.time() > self.last_calibration_time+wait_time
            if enough_corners and enough_time_from_last_cal:

                #opencv can attempt to improve the checkerboard coordinates
                corners = cv2.cornerSubPix(gray, corners, self._conv_size, (-1, -1), self._criteria)

                # store the corners and IDs
                self.corner_loc_img.append(corners)
                self.corner_ids.append(ids)

                # store objective corner positions in a board frame of reference
                board_FOR_corners = self.charuco.board.chessboardCorners[ids, :]
                self.corner_loc_obj.append(board_FOR_corners)
                # 
                self.draw_charuco_outline(corners, ids, self.connected_corners)
                self.last_calibration_time = time.time()
                print(ids)

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

            cv2.line(self._grid_capture_history,point_1, point_2, (255, 165, 0), 1)
 



if __name__ == "__main__":

    charuco = Charuco(4,5,11,8.5,aruco_scale = .75, square_size_overide=.0525, inverted=True)


    capture = cv2.VideoCapture(0)

    print("Getting image size")
    # get the image size:
    image_size = None
    while not image_size:
        read_success, frame = capture.read()

        if read_success:
            image_size = frame.shape
            width = image_size[1]
            height = image_size[0]     

            print(f"Width: {width}  Height: {height}    Size: {image_size}")

 
    detector = IntrinsicCalibrator(charuco, image_size)
    last_calibration_time = time.time()

    print("About to enter main loop")
    while True:
    
        read_success, frame = capture.read()

        detector.collect_corners(frame) 
        #     # add to the calibration corners if enough corners observed
        #     # and enough time  has passed from the last "snapshot"
        #     enough_corners = len(charuco_corner_ids) > min_points_to_process
        #     enough_time_from_last_cal = time.time() > last_calibration_time+time_between_cal

        #     if enough_corners and enough_time_from_last_cal:

        #         #opencv can attempt to improve the checkerboard coordinates
        #         charuco_corners = cv2.cornerSubPix(gray, charuco_corners, conv_size, (-1, -1), criteria)

        #         # store the corners and IDs
        #         self.corner_loc_img.append(charuco_corners)
        #         self.corner_ids.append(charuco_corner_ids)

        #         # objective corner position in a board frame of reference
        #         board_FOR_corners = self.charuco.board.chessboardCorners[charuco_corner_ids, :]
        #         self.corner_loc_obj.append(board_FOR_corners)

        #         # 
        #         self.draw_charuco_outline(charuco_corners, charuco_corner_ids, connected_corners)

        #         last_calibration_time = time.time()

        # # merge calibration footprint and live frame
        # alpha = 1
        # beta = 1
        # merged_frame = cv2.addWeighted(frame, alpha, self.grid_capture_history, beta, 0)
        cv2.imshow("Press 'q' to quit", frame)

        key = cv2.waitKey(1)

        # end capture when enough grids collected
        if key == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            break
 