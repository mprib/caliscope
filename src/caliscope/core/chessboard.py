from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import rtoml


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

    @classmethod
    def from_toml(cls, path: Path) -> "Chessboard":
        """Load Chessboard from TOML file.

        Strips legacy 'square_size_cm' field if present.

        Raises:
            PersistenceError: If file doesn't exist or contains invalid parameters
        """
        from caliscope.persistence import PersistenceError

        if not path.exists():
            raise PersistenceError(f"Chessboard file not found: {path}")

        try:
            data = rtoml.load(path)
            data.pop("square_size_cm", None)  # Strip legacy field
            return cls(**data)
        except Exception as e:
            raise PersistenceError(f"Failed to load Chessboard from {path}: {e}") from e

    def to_toml(self, path: Path) -> None:
        """Save Chessboard to TOML file.

        Raises:
            PersistenceError: If write fails
        """
        from caliscope.persistence import PersistenceError, _safe_write_toml

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _safe_write_toml(asdict(self), path)
        except Exception as e:
            raise PersistenceError(f"Failed to save Chessboard to {path}: {e}") from e

    def get_connected_points(self) -> set[tuple[int, int]]:
        """Point ID pairs that form the grid pattern (adjacent corners only).

        For a rows x columns grid with row-major point IDs, each corner
        connects to its right neighbor and bottom neighbor.
        """
        edges: set[tuple[int, int]] = set()
        for r in range(self.rows):
            for c in range(self.columns):
                point_id = r * self.columns + c
                if c < self.columns - 1:
                    edges.add((point_id, point_id + 1))
                if r < self.rows - 1:
                    edges.add((point_id, point_id + self.columns))
        return edges
