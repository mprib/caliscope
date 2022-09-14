
# %%

import cv2 as cv
import time 
import numpy as np
import time

from collections import defaultdict
from itertools import combinations
class CameraFeeds():
    """
    Create a set of live OpenCV videoCapture devices. These can be then undergo
    individual calibration, stereocalibration, and collect data for 3d
    reconstruction 
    """

    def __init__(self, input_streams,stream_names):
        """
        Initialize input streams. One per camera. Video captures will be created
        for each one during active calibrations and data capture
        """

        self.input_streams = input_streams
        self.stream_names = stream_names
                
        # initialize a number of dictionaries to hold the various parameters 
        # associated with each camera
        self.captures = {}

        # this includes the image size...
        self.image_size = defaultdict(list)

        # the corner locations:
        self.calibration_corners = defaultdict(list)
        
        # the corner IDs:
        self.calibration_ids = defaultdict(list)

        # and the running accumulation of calibration snapshots
        self.calibration_footprint = {}
        self.last_calibration_time = {}

    def collect_calibration_footprint(self, board_threshold, charuco, charuco_inverted=False, time_between_cal=1):
        """
        Charuco: a cv2 charuco board
        board_threshold: percent of board corners that must be represented to record
        
        """
        # build dictionary of all input streams
        for stream_name, strm in zip(self.stream_names, self.input_streams):
            self.captures[stream_name] = cv.VideoCapture(strm)

        # get the pairs of board corners that should be connected:
        connected_corners = get_connected_corners(charuco)

        self.min_points_to_process = int(len(charuco.chessboardCorners) * board_threshold)

        # open the capture streams 
        while True:
            
            # for each stream
            for stream_name, cap in self.captures.items():
                # read in a frame
                read_success, frame = cap.read()

                # initialize parameters on first loop
                if len(self.image_size[stream_name]) == 0 :
                    self.image_size[stream_name] = frame.shape
                    self.calibration_footprint[stream_name] =  np.zeros(frame.shape, dtype='uint8')
                    self.last_calibration_time[stream_name] = time.time()

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
                        cornerColor = (255,25,25))
                   
                    # add to the calibration corners if good read and enough time has passed from last calibration
                    enough_corners = len(charuco_corner_ids) > self.min_points_to_process
                    enough_time_from_last_cal = time.time() > self.last_calibration_time[stream_name]+time_between_cal

                    if enough_corners and enough_time_from_last_cal:
                        self.draw_charuco_outline(stream_name, charuco_corners, charuco_corner_ids, connected_corners)
                        self.calibration_corners[stream_name].append(charuco_corners)
                        self.calibration_ids[stream_name].append(charuco_corner_ids)
                        self.last_calibration_time[stream_name] = time.time()


                # track how many calibrations have been tracked up to now
                # cv.putText(frame, len())

                # merge calibration footprint and live frame
                alpha = 1
                beta = 1 
                merged_frame = cv.addWeighted(frame, alpha, self.calibration_footprint[stream_name], beta, 0)
                cv.imshow(stream_name, merged_frame)

            if cv.waitKey(5) == 27: # ESC to stop   
                break

        self.destroy_captures()

    def destroy_captures(self):
        for nm, cap in self.captures.items():
            cap.release()
        cv.destroyAllWindows()
        self.captures = None

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


    def draw_charuco_outline(self, stream_name, charuco_corners, charuco_ids, connected_corners):
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

            cv.line(self.calibration_footprint[stream_name],point_1, point_2, (255, 165, 0), 1)
   

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



# %%

if __name__ == "__main__":
    feeds = CameraFeeds([0,1], ["Cam_1", "Cam_2"])
    vid_file = 'videos\charuco.mkv'
    # feeds = CameraFeeds([vid_file], ["Cam_1"])
    feeds.collect_calibration_footprint(
        board_threshold=0.7,
        charuco = get_charuco(), 
        charuco_inverted=True,
        time_between_cal=1) # seconds that must pass before new corners are stored


    