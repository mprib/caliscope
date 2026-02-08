from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Chessboard:
    """Chessboard calibration pattern definition.

    Represents the internal corner grid of a chessboard pattern.
    Used for intrinsic calibration only.

    Attributes:
        rows: Number of internal corners vertically (e.g., 6 for 7 rows of squares)
        columns: Number of internal corners horizontally (e.g., 9 for 10 columns of squares)
    """

    rows: int
    columns: int

    def get_object_points(self) -> np.ndarray:
        """Generate 3D object points for all internal corners.

        Points are in row-major order (left-to-right, top-to-bottom when viewing
        the board with the longer edge horizontal). The coordinate frame has:
        - Origin at the top-left internal corner
        - X-axis pointing right
        - Y-axis pointing down
        - Z=0 (planar board)

        Returns:
            Array of shape (rows * columns, 3) with dtype float32.
            Spacing is unit spacing (1.0 between adjacent corners).
        """
        # Standard OpenCV chessboard object points pattern:
        # mgrid[0:columns, 0:rows].T.reshape(-1, 2) produces row-major order
        # (left-to-right, then top-to-bottom) matching findChessboardCorners output.
        object_points = np.zeros((self.rows * self.columns, 3), dtype=np.float32)
        object_points[:, :2] = np.mgrid[0 : self.columns, 0 : self.rows].T.reshape(-1, 2)
        return object_points
