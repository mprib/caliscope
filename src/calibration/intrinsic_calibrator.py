# The purpose of this module will be to construct a worker for the real time 
# device that can look at a picure and identify/collect the corners. 
# There may be a mixed functionality here...I'm not sure. Between the corner
# detector and the corner drawer...like, there will need to be something that
# accumulates a frame of corners to be drawn onto the displayed frame.


import cv2
import time
import numpy as np
from itertools import combinations
import json
import os

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.calibration.charuco import Charuco
from src.cameras.camera import Camera

class IntrinsicCalibrator:

    def __init__(self, camera, charuco):

        self.camera = camera
        self.charuco = charuco

        self.corner_loc_img = []
        self.corner_loc_obj = []
        self.corner_ids = []

        self.connected_corners = self.charuco.get_connected_corners()
        self.last_calibration_time = time.time()    # need to initialize to *something*

        # for subpixel corner correction 
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self._conv_size = (11, 11) # Don't make this too large.

        self.initialize_grid_history()

        self.is_calibrated = False # starts out this way
        
    def initialize_grid_history(self):
        # get appropriately structured image size

        #!!! IF CAMERA RESOLUTION CHANGES THIS MUST BE RERUN
        self.image_size = list(self.camera.resolution)
        self.image_size.reverse()   # for some reason...
        self.image_size.append(3)
        self._grid_capture_history =  np.zeros(self.image_size, dtype='uint8')

        # roll back collected corners to the beginning
        self.corner_loc_img = []
        self.corner_loc_obj = []
        self.corner_ids = []

    def track_corners(self, frame, mirror): 
        """ This method is called by the RealTimeDevice during roll_camera().
        A frame is provided that the IntrinsicCalibrator can then process. This 
        method does the primary work of identifying the corners that are 
        present in the frame."""

        self.frame = frame

        # invert the frame for detection if needed
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        if self.charuco.inverted:
            self.gray = ~ self.gray # invert
        
        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(
            self.gray, 
            self.charuco.board.dictionary)
        
        frame_width = frame.shape[1]
        
        # correct the mirror frame before putting text on it if it's flipped
        if mirror:
            self.frame = cv2.flip(self.frame,1)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 3:
            (success,
            self.charuco_corners, 
            self.charuco_corner_ids) = cv2.aruco.interpolateCornersCharuco(
                aruco_corners,
                aruco_ids,
                self.gray,
                self.charuco.board)
            
            # This occasionally errors out... 
            # only offers possible refinement so if it fails, just move along
            try:
                self.charuco_corners = cv2.cornerSubPix(self.gray, self.charuco_corners, self._conv_size, (-1, -1), self._criteria)
            except:
                pass

            if success:
                # clean up the data types
                self.charuco_corner_ids.tolist()
                self.charuco_corners.tolist()
                
                # flip coordinates if mirrored image fed in
                if mirror:
                    self.charuco_corners[:,:,0] = frame_width-self.charuco_corners[:,:,0]


                for ID, coord in zip(self.charuco_corner_ids[:,0], self.charuco_corners[:,0]):
                    coord = list(coord)
                    # print(frame.shape[1])
                    x = round(coord[0])
                    y = round(coord[1])

                    cv2.circle(self.frame, (x, y), 5,(0,0,220), 3)
                    # cv2.putText(self.frame,str(ID), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5,(220,0,0), 3)

            else:
                self.charuco_corner_ids = np.array([])
                self.charuco_corners = np.array([])
        else:
            self.charuco_corner_ids = np.array([])
            self.charuco_corners = np.array([])

    def collect_corners(self, board_threshold=0.8, wait_time=1):

        corner_count = len(self.charuco.board.chessboardCorners)
        min_points_to_process = int(corner_count * board_threshold)

        if self.charuco_corner_ids.any():
            enough_corners = len(self.charuco_corner_ids) > min_points_to_process
        else:
            enough_corners = False
        
        enough_time_from_last_cal = time.time() > self.last_calibration_time+wait_time
            
        if enough_corners and enough_time_from_last_cal:

            # store the corners and IDs
            self.corner_loc_img.append(self.charuco_corners)
            self.corner_ids.append(self.charuco_corner_ids)

            # store objective corner positions in a board frame of reference
            board_FOR_corners = self.charuco.board.chessboardCorners[self.charuco_corner_ids, :]
            self.corner_loc_obj.append(board_FOR_corners)
            # 
            self.update_capture_history()
            self.last_calibration_time = time.time()

    def update_capture_history(self):
        """
        Given a frame and the location of the charuco board corners within in,
        draw a line connecting the outer bounds of the detected corners and add
        it in to the history of captrued frames. One frame will hold the whole
        history of the corners collected.
        """

        possible_pairs = {pair for pair in combinations(self.charuco_corner_ids.squeeze().tolist(),2)}
        connected_pairs = self.connected_corners.intersection(possible_pairs)

        # build dictionary of corner positions:
        observed_corners = {}
        for crnr_id, crnr in zip(self.charuco_corner_ids.squeeze(), self.charuco_corners.squeeze()):
            observed_corners[crnr_id] = (round(crnr[0]), round(crnr[1]))
        
        # add them to the visual representation of the grid capture history
        for pair in connected_pairs:
            point_1 = observed_corners[pair[0]]
            point_2 = observed_corners[pair[1]]

            cv2.line(self._grid_capture_history,point_1, point_2, (255, 165, 0), 1)

 
    def merged_grid_history(self):
            alpha = 1
            beta = 1

            return cv2.addWeighted(self.frame, alpha, self._grid_capture_history, beta, 0)


    def calibrate(self):
        """
        Use the recorded image corner positions along with the objective
        corner positions based on the board definition to calculated
        the camera matrix and distortion parameters
        """
        print(f"Calibrating....")

        # organize parameters for calibration function
        objpoints = self.corner_loc_obj
        imgpoints = self.corner_loc_img
        height = self.image_size[0]     
        width = self.image_size[1]

        error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, 
            imgpoints, 
            (width, height), 
            None, 
            None)

        self.is_calibrated = True

        # ret is RMSE of reprojection 
        self.camera.error = error
        self.camera.camera_matrix = mtx
        self.camera.distortion = dist
        self.camera.grid_count = len(self.corner_ids)

        print(f"Error: {error}")
        print(f"Camera Matrix: {mtx}")
        print(f"Distortion: {dist}")
        print(f"Grid Count: {self.camera.grid_count}")

    def save_calibration(self, path):
        """
        Store individual camera parameters for use in dual camera calibration
        Saved  to json as camera name to the "calibration_params" directory
        """
        # need to store individual camera parameters

        json_dict = {}
        json_dict["port"] = self.camera.port
        json_dict["image_size"] = self.image_size
        json_dict["camera_matrix"] = self.camera_matrix.tolist()
        json_dict["distortion_params"] = self.distortion_params.tolist()
        json_dict["RMS_reproj_error"] = self.error

        json_object = json.dumps(json_dict, indent=4, separators=(',', ': '))

        with open(os.path.join(Path(__file__).parent, path), "w") as outfile:
                outfile.write(json_object)


if __name__ == "__main__":

    charuco = Charuco(4,5,11,8.5,aruco_scale = .75, square_size_overide=.0525, inverted=True)
    cam = Camera(0)

            
    calib = IntrinsicCalibrator(cam, charuco)
    last_calibration_time = time.time()

    print("About to enter main loop")
    while True:
    
        read_success, frame = cam.capture.read()

        calib.track_corners(frame, mirror=False)
        calib.collect_corners(wait_time=2)
        frame = cv2.flip(frame,1)
        calib.track_corners(frame, mirror=True)
        calib.collect_corners(wait_time=2)
        frame = calib.merged_grid_history() 

        cv2.imshow("Press 'q' to quit", frame)
        # cv2.imshow("Capture History", detector._grid_capture_history)
        key = cv2.waitKey(1)

        # end capture when enough grids collected
        if key == ord('q'):
            cam.capture.release()
            cv2.destroyAllWindows()
            break


    calib.calibrate()
    calib.save_calibration('test_cal.json')

