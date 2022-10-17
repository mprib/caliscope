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

        # get appropriately structured image size
        self.image_size = list(self.camera.resolution)
        self.image_size.reverse()   # for some reason...
        self.image_size.append(3)

        self._grid_capture_history =  np.zeros(self.image_size, dtype='uint8')
        print("Stop here")

    def track_corners(self, frame): 

        self.frame = frame

        # invert the frame for detection if needed
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        if self.charuco.inverted:
            self.gray = ~ self.gray # invert
        
        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(
            self.gray, 
            self.charuco.board.dictionary)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 3:
            _, self.charuco_corners, self.charuco_corner_ids = cv2.aruco.interpolateCornersCharuco(
                aruco_corners,
                aruco_ids,
                self.gray,
                self.charuco.board)

            self.charuco_corners = cv2.cornerSubPix(self.gray, self.charuco_corners, self._conv_size, (-1, -1), self._criteria)

            self.frame = cv2.aruco.drawDetectedCornersCharuco(
                                image = self.frame,
                                charucoCorners=self.charuco_corners,
                                cornerColor = (120,255,0))
        
        else:
            self.charuco_corner_ids = np.array([])
            self.charuco_corners = np.array([])

    def collect_corners(self, board_threshold=0.8, wait_time=1):

        corner_count = len(self.charuco.board.chessboardCorners)
        min_points_to_process = int(corner_count * board_threshold)

        enough_corners = len(self.charuco_corner_ids) > min_points_to_process
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
        draw a line connecting the outer bounds of the detected corners
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
            # frame = cv2.addWeighted(frame, alpha, self._grid_capture_history, beta,0)

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


        # NOTE: ret is RMSE (not sure of what). rvecs and tvecs are the 
        # rotation and translation vectors *for each calibration snapshot*
        # this is, they are the position of the camera relative to the board
        # for that one frame

        self.error = error
        self.camera_matrix = mtx
        self.distortion_params = dist

        print(f"Error: {error}")
        print(f"Camera Matrix: {mtx}")
        print(f"Distortion: {dist}")

    def save_calibration(self, path):
        """
        Store individual camera parameters for use in dual camera calibration
        Saved  to json as camera name to the "calibration_params" directory
        """
        # need to store individual camera parameters

        json_dict = {}
        # json_dict["input_stream"] = self.input_stream
        # json_dict["stream_name"] = self.stream_name
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

        calib.track_corners(frame)
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

