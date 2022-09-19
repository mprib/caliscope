
# %%
import os
import cv2 as cv
import time 
import numpy as np
import time

from itertools import combinations
import json

from pathlib import Path
from charuco import Charuco

class StereoCamera():
    """
    Collect simultaneous calibration data from two cameras to triangulate
    spatial distance of landmarks. Requires previous calibration of an 
    individual Camera object, with calibration parameters saved to json.
    """

    def __init__(self, CamA_name, CamB_name, calibration_folder):
        """
        Initialize parameters from the json files created by the individual
        camera calibration
        """ 

        f = open(Path(Path(__file__).parent, calibration_folder, CamA_name + ".json"))
        self.cam_A_params = json.load(f)
        self.stream_name_A  = self.cam_A_params["stream_name"]
        self.input_stream_A = self.cam_A_params["input_stream"]
        self.cameraMatrix_A = self.cam_A_params["camera_matrix"]
        self.distCoeffs_A = self.cam_A_params["distortion_params"]

        f = open(Path(Path(__file__).parent, calibration_folder, CamB_name + ".json"))
        self.cam_B_params   = json.load(f)
        self.stream_name_B  = self.cam_B_params["stream_name"]
        self.input_stream_B = self.cam_B_params["input_stream"]
        self.cameraMatrix_B = self.cam_B_params["camera_matrix"]
        self.distCoeffs_B = self.cam_B_params["distortion_params"]



    def collect_calibration_corners(self, board_threshold, charuco, charuco_inverted=False, time_between_cal=1):
        """
        This method largely follows the flow of the same named method for
        individual cameras. 

        Charuco: a cv2 charuco board
        board_threshold: percent of board corners that must be represented to record
        """

        # store charuco used for calibration
        self.charuco = charuco

        height = int(1080 * .5)
        width = int(1920 * .5)

        captureA = cv.VideoCapture(self.input_stream_A)
        captureA.set(cv.CAP_PROP_FRAME_WIDTH, width)
        captureA.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        captureB = cv.VideoCapture(self.input_stream_B)
        captureB.set(cv.CAP_PROP_FRAME_WIDTH, width)
        captureB.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        # capture = cv.VideoCapture(self.input_stream)

        min_points_to_process = int(len(self.charuco.board.chessboardCorners) * board_threshold)
        connected_corners = self.charuco.get_connected_corners()

        calibration_initialized = False

        # open the capture stream 
        while True:
            
            # initialize lists of points used for calibration and blank images
            # for the grid overlay
            if not calibration_initialized:
                self.objectpoints  = [] 
                self.imgpointsA = [] 
                self.imgpointsB = []

                # smaple frame to determine actual size
                read_success, frame_A = captureA.read()
                read_success, frame_B = captureB.read()

                self.image_size_A   = frame_A.shape
                self.image_size_B   = frame_B.shape

                self.grid_capture_history_A =  np.zeros(self.image_size_A, dtype='uint8')
                self.grid_capture_history_B =  np.zeros(self.image_size_B, dtype='uint8')

                # for subpixel corner correction 
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.00001)
                conv_size = (11, 11) # Don't make this too large.

                last_calibration_time = time.time()
                calibration_initialized = True

            # read in each frame
            read_success, frame_A = captureA.read()
            read_success, frame_B = captureB.read()
            
            gray_frame_A = cv.cvtColor(frame_A, cv.COLOR_BGR2GRAY)
            gray_frame_B = cv.cvtColor(frame_B, cv.COLOR_BGR2GRAY)

            # check for charuco corners in the image
            found_corner_A, charuco_corners_A, charuco_corner_ids_A = self.find_corners(
                gray_frame_A, 
                charuco_inverted)

            found_corner_B, charuco_corners_B, charuco_corner_ids_B = self.find_corners(
                gray_frame_B, 
                charuco_inverted)
                

            # if charuco corners are detected
            if found_corner_A and found_corner_B:

                # draw them on the frames to visualize
                frame_A = cv.aruco.drawDetectedCornersCharuco(
                    image = frame_A,
                    charucoCorners= charuco_corners_A,
                    cornerColor = (120,255,0))
                
                frame_B = cv.aruco.drawDetectedCornersCharuco(
                    image = frame_B,
                    charucoCorners= charuco_corners_B,
                    cornerColor = (120,255,0))

                # identify the corners that appear in both cameras
                shared_corner_ids = common_corner_ids(charuco_corner_ids_A, charuco_corner_ids_B)
                
                # determine if this snapshot meets the elapsed time and number 
                # of corners criteria
                enough_corners = len(shared_corner_ids) > min_points_to_process
                enough_time_from_last_cal = time.time() > last_calibration_time+time_between_cal

                # proceed if so
                if enough_corners and enough_time_from_last_cal:

                    # identify the charuco corners in a board frame of reference
                    object_points_frame = self.charuco.board.chessboardCorners[shared_corner_ids, :]

                    # improve the checkerboard coordinates
                    charuco_corners_A = cv.cornerSubPix(gray_frame_A, charuco_corners_A, conv_size, (-1, -1), criteria)
                    charuco_corners_B = cv.cornerSubPix(gray_frame_B, charuco_corners_B, conv_size, (-1, -1), criteria)

                    # identify the locations of corners that appear in both frames
                    id_check_A, points_A = common_corner_loc(charuco_corners_A, charuco_corner_ids_A, shared_corner_ids)
                    id_check_B, points_B = common_corner_loc(charuco_corners_B, charuco_corner_ids_B, shared_corner_ids)

                    # add them to the accumulating list of object and image points
                    # checking that all points line up
                    if shared_corner_ids == id_check_A and shared_corner_ids == id_check_B:
                        self.objectpoints.append(object_points_frame)
                        self.imgpointsA.append(np.array(points_A, dtype=np.float32))
                        self.imgpointsB.append(np.array(points_B, dtype=np.float32))

                    # update each grid capture history with the newly added points
                    self.draw_grid_history(points_A, points_B, shared_corner_ids, connected_corners)
                    last_calibration_time = time.time()

            # merge grid history and live frame
            merged_frame_A = cv.addWeighted(frame_A, 1, self.grid_capture_history_A, 1, 0)
            merged_frame_B = cv.addWeighted(frame_B, 1, self.grid_capture_history_B, 1, 0)

            cv.imshow(self.stream_name_A, merged_frame_A)
            cv.imshow(self.stream_name_B, merged_frame_B)


            # end capture when enough grids collected
            if cv.waitKey(5) == 27: # ESC to stop 
                captureA.release()
                captureB.release()
                cv.destroyAllWindows()
                break

    def find_corners(self, gray, charuco_inverted):
        """
        Given a frame, identify the charuco corners in it and return those
        corner locations (x,y) and IDs to the caller
        """
        
        # invert the frame for detection if needed
        if charuco_inverted:
            # gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)  # convert to gray
            gray = ~gray  # invert
        
        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv.aruco.detectMarkers(
            gray, 
            self.charuco.board.dictionary)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 3:
            success, charuco_corners, charuco_corner_ids = cv.aruco.interpolateCornersCharuco(
                aruco_corners,
                aruco_ids,
                gray,
                self.charuco.board)
            
            if charuco_corners is not None:
                return True, charuco_corners, charuco_corner_ids
            else:
                return False, None, None

        else:
            return False, None, None

    def draw_grid_history(self, charuco_corners_A, charuco_corners_B, charuco_ids, connected_corners):
        """
        Given a frame and the location of the shared charuco board corners between
        the two frames, draw a line connecting the outer bounds of the corners 
        they have in common so that they make a grid shape. 
        This drawing is made on the grid capture history for each frame, 
        which is later merged with the frame for visualization
        """

        possible_pairs = {pair for pair in combinations(charuco_ids,2)}
        connected_pairs = connected_corners.intersection(possible_pairs)

        # build dictionary of corner positions for each camera frame:
        observed_corners_A = {}
        for id, crnr in zip(charuco_ids, charuco_corners_A):
            observed_corners_A[id] = (round(crnr[0]), round(crnr[1]))

        observed_corners_B = {}
        for id, crnr in zip(charuco_ids, charuco_corners_B):
            observed_corners_B[id] = (round(crnr[0]), round(crnr[1]))

        # Draw a line for each connected pair on both frames
        # side note: everything about copying and pasting this code feels wrong to me
        for pair in connected_pairs:
            point_1 = observed_corners_A[pair[0]]
            point_2 = observed_corners_A[pair[1]]

            cv.line(self.grid_capture_history_A,point_1, point_2, (255, 165, 0), 1)
            
            point_1 = observed_corners_B[pair[0]]
            point_2 = observed_corners_B[pair[1]]
            cv.line(self.grid_capture_history_B,point_1, point_2, (255, 165, 0), 1)


    def calibrate(self):
        """
        generate rotation and translation vectors relating position of Camera B
        relative to Camera A
        """

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.000001)

        height = self.image_size_A[0]    
        width = self.image_size_A[1]

        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
            objectPoints = self.objectpoints,
            imagePoints1 = self.imgpointsA,
            imagePoints2 = self.imgpointsB,
            cameraMatrix1 = np.array(self.cameraMatrix_A),
            distCoeffs1 = np.array(self.distCoeffs_A),
            cameraMatrix2 = np.array(self.cameraMatrix_B),
            distCoeffs2 = np.array(self.distCoeffs_B),

            # based on https://stackoverflow.com/questions/35128281/different-image-size-opencv-stereocalibrate
            # image size does not matter given the approach used here...not sure
            imageSize = (height, width), 
            flags = cv.CALIB_FIX_INTRINSIC, # this is the default; only R, T, E, and F matrices are estimated.
            criteria = criteria) 
        
        self.rotation_AB = R
        self.translation_AB = T
        print(f"Error: {ret}")
        print(f"Rotation: {R}")
        print(f"Translation: {T}")

    def save_calibration(self, destination_folder):
        """
        Store individual camera parameters for use in dual camera calibration
        Saved  to json as camera name to the "calibration_params" directory
        """
        # need to store individual camera parameters

        json_dict = {}




        json_dict["input_stream"] = self.input_stream
        json_dict["stream_name"] = self.stream_name
        json_dict["image_size"] = self.image_size
        json_dict["camera_matrix"] = self.camera_matrix.tolist()
        json_dict["distortion_params"] = self.distortion_params.tolist()

        json_object = json.dumps(json_dict, indent=4, separators=(',', ': '))

        with open(os.path.join(Path(__file__).parent, destination_folder + "/" + self.stream_name + ".json"), "w") as outfile:
            outfile.write(json_object)


###################### HELPER FUNCTIONS ########################################

def common_corner_ids(IDs_A, IDs_B):
    """
    Properly format a list of corner IDs that are shared between the two frames
    where the inputs are arrays
    """

    # reduce unwanted dimensions
    IDs_A = [i[0] for i in IDs_A]
    IDs_B = [i[0] for i in IDs_B]

    # return the shared corners between them

    return list(set(IDs_A) & set(IDs_B))

def common_corner_loc(corners, ids, shared_ids):
    """
    Given a list of corner positions and ids, return position of only the 
    subset of shared IDs. ID_Check is returned to allow confirmation that
    the order of the image points remains the same
    """

    id_check = []
    cc = []

    for id, corner in zip(ids, corners):
        if id in shared_ids:
            id_check.append(id[0])
            cc.append(corner[0].tolist())
        
    return id_check, cc

# %%

# stereocam = StereoCamera("cam_0", "cam_1", "calibration_params")
# charuco = Charuco(4,5,11,8.5)
# stereocam.read_json("calibration_params", "test_stereocal")

# %%
# stereocam.write_json("calibration_params", "test_stereocal")

# %%

# %%
# stereocam.calibrate()



if __name__ == "__main__":
    stereocam = StereoCamera("cam_0", "cam_1", "calibration_params")

    charuco = Charuco(4,5,11,8.5, aruco_scale=.75, square_size_overide=.0525)
    # charuco.save_image("src/new_charuco.png")


    stereocam.collect_calibration_corners(
        board_threshold=0.4,
        charuco = charuco, 
        charuco_inverted=True,
        time_between_cal=.5)

    # stereocam.write_json("calibration_params", "test_stereocal")
    
    stereocam.calibrate()