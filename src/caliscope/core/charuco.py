# %%
from __future__ import annotations

# NOTE: Conversions are being made here between inches and cm because
# this seems like a reasonable scale for discussing the board, but when
# it is actually created in OpenCV, the board height is expressed
# in meters as a standard convention of science, and to improve
# readability of 3D positional output downstream

import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import rtoml

logger = logging.getLogger(__name__)

INCHES_PER_CM = 0.393701

# generateImage aborts on a quasi-periodic set of output sizes (an integer-pixel
# rounding artifact). board_img retries upward through that band; this caps the
# search so a genuinely unrenderable board fails loudly instead of looping.
MAX_RENDER_RETRIES = 64

# Standard ArUco families come in these pool sizes, ordered smallest-first.
# The pools are nested (the 50-pool is the first 50 markers of the 250-pool,
# etc.), so enlarging the pool never changes the markers an existing board uses.
_STANDARD_POOL_LADDER = (50, 100, 250, 1000)


class DictionaryCapacityError(ValueError):
    """Raised when a board needs more markers than its dictionary family holds.

    A ChArUco board places one marker per white square. If the board needs more
    markers than the largest pool in its family, no rendering resolution can save
    it: the marker ids overflow the dictionary's bytesList. Subclasses ValueError
    so a caller doing broad bad-config handling still catches it.
    """

    def __init__(self, dictionary: str, needed: int, capacity: int) -> None:
        self.dictionary = dictionary
        self.needed = needed
        self.capacity = capacity
        super().__init__(f"board needs {needed} markers; {dictionary} holds {capacity}.")


def fit_dictionary_pool(dictionary: str, marker_count: int) -> str:
    """Return the smallest pool in `dictionary`'s family that holds marker_count.

    For laddered families (DICT_4X4/5X5/6X6/7X7), parses the family prefix and
    returns the smallest of 50/100/250/1000 whose real pool size (checked against
    `len(bytesList)`, never the assumed ladder value) holds marker_count. The
    pools are nested, so changing the pool either way (grow or shrink) never
    changes the markers an existing board uses.

    Non-laddered dictionaries (DICT_ARUCO_ORIGINAL, DICT_APRILTAG_*) have no
    family ladder and are returned unchanged.

    Raises DictionaryCapacityError if no pool in the family is large enough.
    """
    family, _, pool = dictionary.rpartition("_")
    if not pool.isdigit() or int(pool) not in _STANDARD_POOL_LADDER:
        # Not a laddered standard family (e.g. DICT_ARUCO_ORIGINAL, DICT_APRILTAG_16h5).
        # Validate capacity against the real pool and leave the value alone.
        capacity = len(cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary]).bytesList)
        if marker_count > capacity:
            raise DictionaryCapacityError(dictionary, marker_count, capacity)
        return dictionary

    for candidate_pool in _STANDARD_POOL_LADDER:
        candidate = f"{family}_{candidate_pool}"
        capacity = len(cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[candidate]).bytesList)
        if marker_count <= capacity:
            return candidate

    largest = f"{family}_{_STANDARD_POOL_LADDER[-1]}"
    capacity = len(cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[largest]).bytesList)
    raise DictionaryCapacityError(largest, marker_count, capacity)


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
        square_size_override_cm=None,
        inverted=False,
        legacy_pattern=False,
    ):  # after printing, measure actual and return to override
        """
        Create board based on shape and dimensions
        square_size_override_cm: correct for the actual printed size of the board
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
        self.square_size_override_cm = square_size_override_cm
        self.inverted = inverted
        self.legacy_pattern = legacy_pattern

    @classmethod
    def from_squares(
        cls,
        columns: int,
        rows: int,
        square_size_cm: float,
        *,
        dictionary: str = "DICT_4X4_50",
        aruco_scale: float = 0.75,
        inverted: bool = False,
        legacy_pattern: bool = False,
    ) -> Charuco:
        """Create a Charuco board from grid dimensions and square size.

        Args:
            square_size_cm: Edge length of each square in centimeters.
                This determines the scale of calibrated 3D coordinates,
                which will be in meters (e.g., 3.0 cm squares produce
                corners spaced 0.03 m apart in object space).
                Post-alignment WorldPoints and TRC exports are in meters.

        Example:
            >>> charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
        """
        board_height_cm = rows * square_size_cm
        board_width_cm = columns * square_size_cm

        return cls(
            columns=columns,
            rows=rows,
            board_height=board_height_cm,
            board_width=board_width_cm,
            dictionary=dictionary,
            units="cm",
            aruco_scale=aruco_scale,
            square_size_override_cm=square_size_cm,
            inverted=inverted,
            legacy_pattern=legacy_pattern,
        )

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
        if self.square_size_override_cm:
            square_length = self.square_size_override_cm / 100  # note: in cm within GUI
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
        """Render the board as a cv2 image, retrying upward until generateImage succeeds.

        generateImage aborts on a quasi-periodic set of output sizes (an
        integer-pixel rounding artifact, not just small sizes), so no single
        scale is provably safe. Render at the requested scale and nudge it
        upward one pixel at a time until OpenCV accepts it.

        A larger pixmap_scale yields a printer-ready png.
        """
        board = self.board
        scale = int(pixmap_scale)
        last_error = None
        for _ in range(MAX_RENDER_RETRIES):
            try:
                img = board.generateImage(
                    (self.board_width_scaled(pixmap_scale=scale), self.board_height_scaled(pixmap_scale=scale))
                )
                break
            except cv2.error as e:
                # Keep the cause: if this isn't the resolution artifact (e.g. a
                # mismatched dictionary), incrementing won't help and the real
                # OpenCV message should survive in the chained traceback.
                last_error = e
                scale += 1
        else:
            raise RuntimeError(
                f"No renderable scale for {self.columns}x{self.rows} board "
                f"within {MAX_RENDER_RETRIES} steps of {pixmap_scale}"
            ) from last_error

        if self.inverted:
            img = cv2.bitwise_not(img)

        return img

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

    def get_connected_points(self) -> set[tuple[int, int]]:
        """
        For a given board, returns a set of corner id pairs that will connect to form
        a grid pattern. This will provide the "object points" used by the calibration
        functions. It is the ground truth of how the points relate in the world.

        The return value is a *set* not a list
        """
        # create sets of the vertical and horizontal line positions
        # getChessboardCorners returns MatLike; convert to ndarray for indexing
        corners = np.asarray(self.board.getChessboardCorners())
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
        corners = np.asarray(self.board.getChessboardCorners())
        return corners[corner_ids, :]

    @property
    def marker_count(self) -> int:
        """Number of ArUco markers the board uses (one per white square).

        Markers sit on alternating squares, so the count is (columns * rows) // 2.
        Verified equal to OpenCV's board.getIds() across sizes and both legacy
        modes; computing it directly avoids constructing a board (which would need
        a dictionary — the very thing fit_dictionary_pool is choosing).
        """
        return (self.columns * self.rows) // 2

    def _fit_dictionary(self) -> None:
        """Normalize self.dictionary to the smallest pool that holds the board's markers.

        Grows an undersized pool and shrinks an oversized one, since nested pools
        make both safe. Applied at the persistence boundary so the stored value
        always names the dictionary actually used. Raises DictionaryCapacityError
        if no pool fits.
        """
        self.dictionary = fit_dictionary_pool(self.dictionary, self.marker_count)

    @classmethod
    def from_toml(cls, path: Path) -> "Charuco":
        """Load Charuco board definition from TOML file.

        Normalizes the dictionary pool to fit the board (see fit_dictionary_pool),
        correcting a hand-edited TOML on load.

        Raises:
            PersistenceError: If file doesn't exist or contains invalid parameters
            DictionaryCapacityError: If the board needs more markers than its
                dictionary family can hold
        """
        from caliscope.persistence import PersistenceError

        if not path.exists():
            raise PersistenceError(f"Charuco file not found: {path}")

        try:
            data = rtoml.load(path)
            charuco = cls(**data)
        except Exception as e:
            raise PersistenceError(f"Failed to load Charuco from {path}: {e}") from e

        charuco._fit_dictionary()
        return charuco

    def to_toml(self, path: Path) -> None:
        """Save Charuco board definition to TOML file.

        Enumerates fields explicitly rather than using __dict__ to avoid
        serializing computed properties or internal state. Normalizes the
        dictionary pool to fit the board (see fit_dictionary_pool) so the stored
        value always names the dictionary actually used; note this mutates
        self.dictionary in place. The fit is idempotent, so if a caller holds a
        pre-fit copy it reconverges to the same value on its next save.

        Raises:
            PersistenceError: If write fails
            DictionaryCapacityError: If the board needs more markers than its
                dictionary family can hold
        """
        from caliscope.persistence import PersistenceError, _safe_write_toml

        self._fit_dictionary()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "columns": self.columns,
                "rows": self.rows,
                "board_height": self.board_height,
                "board_width": self.board_width,
                "dictionary": self.dictionary,
                "units": self.units,
                "aruco_scale": self.aruco_scale,
                "square_size_override_cm": self.square_size_override_cm,
                "inverted": self.inverted,
                "legacy_pattern": self.legacy_pattern,
            }
            # Filter None values to prevent rtoml "null" strings
            clean_data = {k: v for k, v in data.items() if v is not None}
            _safe_write_toml(clean_data, path)
        except Exception as e:
            raise PersistenceError(f"Failed to save Charuco to {path}: {e}") from e

    def summary(self):
        text = f"Columns: {self.columns}\n"
        text = text + f"Rows: {self.rows}\n"
        text = text + f"Board Size: {self.board_width} x {self.board_height} {self.units}\n"
        text = text + f"Inverted:  {self.inverted}\n"
        text = text + "\n"
        text = text + f"Square Edge Length: {self.square_size_override_cm} cm"
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
    charuco = Charuco(4, 5, 4, 8.5, aruco_scale=0.75, units="inch", inverted=True, square_size_override_cm=5.25)
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
