
# %%

from logging import exception
import cv2 as cv
import time 
import numpy as np

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
        
        #TODO: Figure out how to use exceptions
        # if len(stream_names)!=len(input_streams):
        # also, stream_names must all be unique    

        self.input_streams = input_streams
        self.stream_names = stream_names
                


    def calibrate(self, board_threshold, charuco, charuco_inverted=False):
        """
        Charuco: a cv2 charuco board
        board_threshold: percent of board corners that must be represented to record
        
        """
        # build dictionary of all input streams
        self.captures = {}
        for stream_name, strm in zip(self.stream_names, self.input_streams):
            self.captures[stream_name] = cv.VideoCapture(strm)

        # and a place to record the image size
        image_size = {}
        for stream_name in self.stream_names:
            image_size[stream_name] = None


        # get a list of all the board corners that should be connected:
        connected_corners = get_connected_corners(charuco)

        self.min_points_to_process = int(len(charuco.chessboardCorners) * board_threshold)

        # open the capture streams 
        while True:
            
            # for each stream
            for stream_name, cap in self.captures.items():
                # read in a frame
                read_success, frame = cap.read()

                # set the image size if unknown
                if image_size[stream_name] is None:
                    image_size[stream_name] = frame.shape
                    self.calibration_footprint = {}
                    self.calibration_footprint[stream_name] =  np.zeros(frame.shape, dtype='uint8')

                # check for charuco corners in the image
                found_corner, charuco_corners, charuco_corner_ids = self.find_corners(
                    frame, 
                    charuco, 
                    charuco_inverted)


                # if charuco corners are detected
                if found_corner:

                    # draw them on the frame to visualize
                    # frame = cv.aruco.drawDetectedCornersCharuco(
                    #     image = frame,
                    #     charucoCorners=charuco_corners,
                    #     charucoIds=charuco_corner_ids,
                    #     cornerColor = (255,25,25))
                   
                    # draw a box bounding each of the frames
                    if len(charuco_corner_ids) > self.min_points_to_process:
                        frame = self.calibration_footprint[stream_name]
                        frame = self.drawCharucoOutline(frame, charuco_corners, charuco_corner_ids, connected_corners)

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
        if len(aruco_corners) > 0:
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



    def update_calibration_corners(self, corners, corner_ids):
        pass


        # return 


    def drawCharucoOutline(self, frame, charuco_corners, charuco_ids, connected_corners):
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
        
        for pair in connected_pairs:
            point_1 = observed_corners[pair[0]]
            point_2 = observed_corners[pair[1]]

            cv.line(frame,point_1, point_2, (255,255,255), 3)
        
        return frame



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
    a grid pattern
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
c_c = get_connected_corners(get_charuco())

print(c_c)

# %%

if __name__ == "__main__":
    # feeds = CameraFeeds([0,1], ["Cam_1", "Cam_2"])
    vid_file = 'videos\charuco.mkv'
    feeds = CameraFeeds([vid_file], ["Cam_1"])
    feeds.calibrate(
        board_threshold=0.7,
        charuco = get_charuco(), 
        charuco_inverted=True)