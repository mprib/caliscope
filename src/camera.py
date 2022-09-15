
# %%

import cv2 as cv
import time 
import numpy as np
import time

from collections import defaultdict
from itertools import combinations


class Camera():
    """
    Create a set of live OpenCV videoCapture devices. These can be then undergo
    individual calibration, stereocalibration, and collect data for 3d
    reconstruction 
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


    def collect_calibration_corners(self, board_threshold, charuco, charuco_inverted=False, time_between_cal=1):
        """
        Charuco: a cv2 charuco board
        board_threshold: percent of board corners that must be represented to record
        """

        capture = cv.VideoCapture(self.input_stream)

        min_points_to_process = int(len(charuco.chessboardCorners) * board_threshold)
        connected_corners = get_connected_corners(charuco)

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
                charuco, 
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
                break

        # clean up
        capture.release()
        cv.destroyAllWindows()

    
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

        cv.calibrateCameraExtended
            

##############################      CLASS ENDS     ###################################
# Helper functions here primarily related to managing the charuco
# may consider organizing as a charuco class

def get_charuco():
    # get aruco marker dictionary
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

    # and the size of the target board in real life
    charuco_height_inch = 11 # inches
    charuco_width_inch = 8.5 # inches

    # convert to meters
    charuco_height = charuco_height_inch/39.37
    charuco_width = charuco_width_inch/39.37

    # assign the board layout
    charuco_columns = 4
    charuco_rows = 5

    # set the square length to maximize the paper
    square_length = min([charuco_height/charuco_rows, 
                        charuco_width/charuco_columns]) 

    # while making the aruco large to improve visability
    aruco_length = square_length * 0.9 

    # create the board
    board = cv.aruco.CharucoBoard_create(charuco_columns, charuco_rows, square_length, aruco_length, dictionary)

    return board


# %%
def get_connected_corners(board):
    """
    For a given board, returns a set of corner id pairs that will connect to form
    a grid pattern.

    NOTE: the return value is a *set* not a list
    """
    # create sets of the vertical and horizontal line positions
    corners = board.chessboardCorners
    corners_x = corners[:,0]
    corners_y = corners[:,1]
    x_set = set(corners_x)
    y_set = set(corners_y)


    lines = defaultdict(list)

    # put each point on the same vertical line in a list
    for x_line in x_set:
        for corner, x, y in zip(range(0, len(corners)), corners_x, corners_y):
            if x == x_line:
                lines[f"x_{x_line}"].append(corner)

    # and the same for each point on the same horizontal line
    for y_line in y_set:
        for corner, x, y in zip(range(0, len(corners)), corners_x, corners_y):
            if y == y_line:
                lines[f"y_{y_line}"].append(corner)

    # create a set of all sets of corner pairs that should be connected
    connected_corners = set()
    for lines, corner_ids in lines.items():
        for i in combinations(corner_ids, 2):
            connected_corners.add(i)

    return connected_corners


# def get_object_points(board, corner_ids):
# 
    # for id in corner_ids



# %%

if __name__ == "__main__":
    feeds = {0: "Cam_1",1:"Cam_2"}
    vid_file = 'videos\charuco.mkv'
    
    for stream, stream_name in feeds.items():
        active_camera = Camera(stream, stream_name)
        active_camera.collect_calibration_corners(
            board_threshold=0.7,
            charuco = get_charuco(), 
            charuco_inverted=True,
            time_between_cal=1) # seconds that must pass before new corners are stored

        active_camera.calibrate()


    