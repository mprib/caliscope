from email.errors import InvalidBase64CharactersDefect
import cv2 as cv
from collections import defaultdict
from itertools import combinations



class Charuco():
    """
    create a charuco board that can be printed out and used for camera 
    calibration, and used for drawing a grid during calibration    
    """

    def __init__(
        self, 
        columns, 
        rows, 
        board_height, 
        board_width, 
        dictionary="DICT_4X4_50",
        units="inches", 
        aruco_scale=0.75, 
        square_size_overide=None):
        
        """
        Create board based on shape and dimensions

        square_size_overide: correct for the actual printed size of the board
        """
        self.columns = columns
        self.rows = rows

        if units == "inches":
            # convert to meters
            board_height = board_height/39.37
            board_width = board_width/39.37

        self.board_height = board_height
        self.board_width = board_width

        # if square length not provided, calculate based on board dimensions
        # to maximize size of squares
        if square_size_overide:
            square_length = square_size_overide # note: in meters
        else:
            square_length = min([board_height/rows, board_width/columns]) 

        # Scale aruco according based on square size
        aruco_length = square_length * aruco_scale 

        dictionary_integer = ARUCO_DICTIONARIES[dictionary]
        self.dictionary = cv.aruco.Dictionary_get(dictionary_integer)

        self.board = cv.aruco.CharucoBoard_create(
            columns,
            rows,
            square_length,
            aruco_length,
            self.dictionary)
        


    def save_image(self, path, inverted=True):

        # convert to inches for ease of saving at 300 DPI
        inches_per_meter = 39.37

        width_inch = self.board_width * inches_per_meter
        height_inch = self.board_height * inches_per_meter

        charuco_img = self.board.draw((int(width_inch*300), int(height_inch*300)))

        if inverted:
            cv.imwrite(path, ~charuco_img)
        else:
            cv.imwrite(path, charuco_img)

        # self.board.

    def get_connected_corners(self):
        """
        For a given board, returns a set of corner id pairs that will connect to form
        a grid pattern.

        NOTE: the return value is a *set* not a list
        """
        # create sets of the vertical and horizontal line positions
        corners = self.board.chessboardCorners
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



    def get_object_corners(self, corner_ids):
        """
        Given an array of corner IDs, provide an array of their relative 
        position in a board from of reference, originating from a corner position.
        """

        return self.board.chessboardCorners[corner_ids, :]


 


################################## REFERENCE ###################################
ARUCO_DICTIONARIES = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}
