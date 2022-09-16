
# %%
import cv2 as cv
import time 
import numpy as np
import time

from itertools import combinations
import json

from pathlib import Path

class StereoCamera():
    """
    Collect simultaneous calibration data from two cameras to triangulate
    spatial distance of landmarks. Requires previous calibration of an 
    individual Camera object, with calibration parameters saved to json.
    """

    def __init__(self, CamA_name, CamB_name, calibration_folder):
        
        # NOTE:I'm on the fence about whether it's good policy to just
        # store these parameters as 
        f = open(Path(calibration_folder, CamA_name + ".json"))
        self.cam_A_params = json.load(f)
        self.stream_name_A = self.cam_A_params["stream_name"]
        self.image_size_A = self.cam_A_params["image_size"]

        f = open(Path(calibration_folder, CamB_name + ".json"))
        self.cam_B_params = json.load(f)
        self.stream_name_B = self.cam_B_params["stream_name"]
        self.image_size_B = self.cam_B_params["image_size"]


    def collect_calibration_corners(self, board_threshold, charuco, charuco_inverted=False, time_between_cal=1):
        """
        This method largely follows the flow of the same named method for
        individual cameras. 

        Charuco: a cv2 charuco board
        board_threshold: percent of board corners that must be represented to record
        """

        # store charuco used for calibration
        self.charuco = charuco

        captureA = cv.VideoCapture(self.cam_A_params["stream_name"])
        captureB = cv.VideoCapture(self.cam_B_params["stream_name"])
        # capture = cv.VideoCapture(self.input_stream)

        min_points_to_process = int(len(self.charuco.board.chessboardCorners) * board_threshold)
        connected_corners = self.charuco.get_connected_corners()

        # open the capture stream 
        while True:
        
            read_success, frame_A = captureA.read()
            read_success, frame_B = captureB.read()
            
            # initialize parameters on first loop
            if len(self.image_size) == 0 :
                self.grid_capture_history_A =  np.zeros(self.image_size_A, dtype='uint8')
                self.grid_capture_history_B =  np.zeros(self.image_size_B, dtype='uint8')
                last_calibration_time = time.time()

            # check for charuco corners in the image
            found_corner_A, charuco_corners_A, charuco_corner_ids = self.find_corners(
                frame_A, 
                charuco_inverted)

            found_corner_B, charuco_corners_B, charuco_corner_ids = self.find_corners(
                frame_B, 
                charuco_inverted)
                

            # if charuco corners are detected
            if found_corner_A:
                # TODO: MAC: THIS IS WHERE YOU CALLED IT A DAY..
                # START HERE TOMORROW!
                
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

    def find_corners(self, frame, charuco_inverted):
        """
        Given a frame, identify the charuco corners in it and return those
        corners and IDs to the caller
        """
        
        # invert the frame for detection if needed
        if charuco_inverted:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert to gray
            frame = ~frame  # invert
        
        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv.aruco.detectMarkers(
            frame, 
            self.charuco.board.dictionary)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 3:
            success, charuco_corners, charuco_corner_ids = cv.aruco.interpolateCornersCharuco(
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
# The code below is the ultimate goal here...
# stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
# ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
# mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
# %%
