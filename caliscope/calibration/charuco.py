# %%

# NOTE: Conversions are being made here between inches and cm because
# this seems like a reasonable scale for discussing the board, but when
# it is actually created in OpenCV, the board height is expressed
# in meters as a standard convention of science, and to improve
# readability of 3D positional output downstream

from collections import defaultdict
from itertools import combinations

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

import caliscope.logger

logger = caliscope.logger.get(__name__)

INCHES_PER_CM = 0.393701


class Charuco:
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
        units="inch",
        aruco_scale=0.75,
        square_size_overide_cm=None,
        inverted=False,
        legacy_pattern=False,
    ):  # after printing, measure actual and return to overide
        """
        Create board based on shape and dimensions
        square_size_overide_cm: correct for the actual printed size of the board
        """
        self.columns = columns
        self.rows = rows

        self.board_height = board_height
        self.board_width = board_width
        self.dictionary = dictionary

        self.units = units
        self.aruco_scale = aruco_scale
        # if square length not provided, calculate based on board dimensions
        # to maximize size of squares
        self.square_size_overide_cm = square_size_overide_cm
        self.inverted = inverted
        self.legacy_pattern = legacy_pattern

    @property
    def board_height_cm(self):
        """Internal calculations will always use mm for consistency"""
        if self.units == "inch":
            return self.board_height / INCHES_PER_CM
        else:
            return self.board_height

    @property
    def board_width_cm(self):
        """Internal calculations will always use mm for consistency"""
        if self.units == "inch":
            return self.board_width / INCHES_PER_CM
        else:
            return self.board_width

    def board_height_scaled(self, pixmap_scale):
        if self.board_height_cm > self.board_width_cm:
            scaled_height = int(pixmap_scale)
        else:
            scaled_height = int(pixmap_scale * (self.board_height_cm / self.board_width_cm))
        return scaled_height

    def board_width_scaled(self, pixmap_scale):
        if self.board_height_cm > self.board_width_cm:
            scaled_width = int(pixmap_scale * (self.board_width_cm / self.board_height_cm))
        else:
            scaled_width = int(pixmap_scale)

        return scaled_width

    @property
    def dictionary_object(self):
        # grab the dictionary from the reference info at the foot of the module
        dictionary_integer = ARUCO_DICTIONARIES[self.dictionary]
        return cv2.aruco.getPredefinedDictionary(dictionary_integer)

    @property
    def board(self):
        if self.square_size_overide_cm:
            square_length = self.square_size_overide_cm / 100  # note: in cm within GUI
        else:
            board_height_m = self.board_height_cm / 100
            board_width_m = self.board_width_cm / 100

            square_length = min([board_height_m / self.rows, board_width_m / self.columns])
        logger.info(f"Creating charuco with square length of {round(square_length, 4)}")

        aruco_length = square_length * self.aruco_scale
        # create the board
        board = cv2.aruco.CharucoBoard(
            size=(self.columns, self.rows),
            squareLength=square_length,
            markerLength=aruco_length,
            dictionary=self.dictionary_object,
        )

        logger.info(f"Setting legacy pattern of board to {self.legacy_pattern}")
        board.setLegacyPattern(self.legacy_pattern)
        return board

    def board_img(self, pixmap_scale=1000):
        """
        returns a cv2 image (numpy array) of the board
        smaller scale image by default for display to GUI
        provide larger max_edge_length to get printer-ready png
        """
        img = self.board.generateImage(
            (self.board_width_scaled(pixmap_scale=pixmap_scale), self.board_height_scaled(pixmap_scale=pixmap_scale))
        )
        if self.inverted:
            img = ~img

        return img

    def board_pixmap(self, width, height):
        """
        Convert from an opencv image to QPixmap
        this can be used for creating thumbnail images
        """
        rgb_image = cv2.cvtColor(self.board_img(), cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        charuco_QImage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = charuco_QImage.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QPixmap.fromImage(p)

    def save_image(self, path):
        """
        Saving image at 10x higher resolution than used for GUI
        """
        cv2.imwrite(path, self.board_img(pixmap_scale=10000))

    def save_mirror_image(self, path):
        """
        Saving image at 10x higher resolution than used for GUI
        """
        mirror = cv2.flip(self.board_img(pixmap_scale=10000), 1)
        cv2.imwrite(path, mirror)

    def get_connected_points(self):
        """
        For a given board, returns a set of corner id pairs that will connect to form
        a grid pattern. This will provide the "object points" used by the calibration
        functions. It is the ground truth of how the points relate in the world.

        The return value is a *set* not a list
        """
        # create sets of the vertical and horizontal line positions
        corners = self.board.getChessboardCorners()
        corners_x = corners[:, 0]
        corners_y = corners[:, 1]
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
        position in a board frame of reference, originating from a corner position.
        """

        return self.board.chessboardCorners()[corner_ids, :]

    def summary(self):
        text = f"Columns: {self.columns}\n"
        text = text + f"Rows: {self.rows}\n"
        text = text + f"Board Size: {self.board_width} x {self.board_height} {self.units}\n"
        text = text + f"Inverted:  {self.inverted}\n"
        text = text + "\n"
        text = text + f"Square Edge Length: {self.square_size_overide_cm} cm"
        return text


################################## REFERENCE ###################################
ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


if __name__ == "__main__":
    charuco = Charuco(4, 5, 4, 8.5, aruco_scale=0.75, units="inch", inverted=True, square_size_overide_cm=5.25)
    charuco.save_image("test_charuco.png")
    width, height = charuco.board_img().shape
    logger.info(f"Board width is {width}\nBoard height is {height}")

    corners = charuco.board.getChessboardCorners()
    logger.info(corners)

    logger.info(f"Charuco dictionary: {charuco.__dict__}")
    # while True:
    #     cv2.imshow("Charuco Board...'q' to quit", charuco.board_img)
    #     #
    #     key = cv2.waitKey(0)
    #     if key == ord("q"):
    #         cv2.destroyAllWindows()
    #         break

# %%
