from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import rtoml
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ArucoTarget:
    """A rigid calibration target with ArUco markers at known 3D positions.

    Used for extrinsic calibration. The target defines a coordinate frame
    where markers have known corner positions (in meters).
    """

    dictionary: int
    corners: dict[int, NDArray[np.float64]]  # marker_id -> (4, 3) positions
    marker_size_m: float

    @staticmethod
    def single_marker(
        marker_id: int = 0,
        marker_size_m: float = 0.05,
        dictionary: int = cv2.aruco.DICT_4X4_100,
    ) -> "ArucoTarget":
        """Factory for single-marker target (most common case).

        Creates a target with one marker centered at origin. Corner positions
        follow OpenCV's ArUco convention: origin at center, X right, Y up, Z out.
        Corners ordered TL, TR, BR, BL as returned by detectMarkers.
        """
        s = marker_size_m / 2
        # OpenCV ArUco convention: Y points UP (standard math frame, not image frame)
        corner_positions = np.array(
            [
                [-s, +s, 0.0],  # TL
                [+s, +s, 0.0],  # TR
                [+s, -s, 0.0],  # BR
                [-s, -s, 0.0],  # BL
            ],
            dtype=np.float64,
        )

        return ArucoTarget(
            dictionary=dictionary,
            corners={marker_id: corner_positions},
            marker_size_m=marker_size_m,
        )

    @property
    def marker_ids(self) -> list[int]:
        """All marker IDs this target tracks, sorted."""
        return sorted(self.corners.keys())

    @classmethod
    def from_toml(cls, path: Path) -> "ArucoTarget":
        """Load ArucoTarget from TOML file.

        TOML format:
            dictionary = 0
            marker_size_m = 0.05
            [corners.0]
            positions = [[-0.025, -0.025, 0.0], [0.025, -0.025, 0.0], ...]

        Raises:
            PersistenceError: If file doesn't exist or format is invalid
        """
        from caliscope.persistence import PersistenceError

        if not path.exists():
            raise PersistenceError(f"ArucoTarget file not found: {path}")

        try:
            data = rtoml.load(path)

            dictionary = data["dictionary"]
            marker_size_m = data["marker_size_m"]

            corners: dict[int, NDArray[np.float64]] = {}
            for marker_id_str, corner_data in data.get("corners", {}).items():
                marker_id = int(marker_id_str)
                positions = np.array(corner_data["positions"], dtype=np.float64)
                if positions.shape != (4, 3):
                    raise ValueError(f"Marker {marker_id} has invalid shape: {positions.shape}")
                corners[marker_id] = positions

            return cls(
                dictionary=dictionary,
                corners=corners,
                marker_size_m=marker_size_m,
            )
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to load ArucoTarget from {path}: {e}") from e

    def to_toml(self, path: Path) -> None:
        """Save ArucoTarget to TOML file.

        Raises:
            PersistenceError: If write fails
        """
        from caliscope.persistence import PersistenceError, _safe_write_toml

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            corners_data = {}
            for marker_id, positions in self.corners.items():
                corners_data[str(marker_id)] = {"positions": positions.tolist()}

            data = {
                "dictionary": self.dictionary,
                "marker_size_m": self.marker_size_m,
                "corners": corners_data,
            }

            _safe_write_toml(data, path)
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to save ArucoTarget to {path}: {e}") from e

    def get_corner_positions(self, marker_id: int) -> NDArray[np.float64]:
        """Get (4, 3) corner positions for a marker.

        Raises:
            KeyError: If marker_id not in this target
        """
        return self.corners[marker_id]

    def generate_marker_image(self, marker_id: int, pixels_per_meter: int = 4000) -> NDArray:
        """Generate annotated printable marker image.

        All annotations are in the white border. A small axis legend in the
        bottom-right shows the coordinate frame orientation without occluding
        the marker pattern.

        Raises:
            KeyError: If marker_id not in this target
        """
        if marker_id not in self.corners:
            raise KeyError(f"Marker {marker_id} not in target (available: {self.marker_ids})")

        pixel_size = int(self.marker_size_m * pixels_per_meter)
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, pixel_size)

        border = pixel_size // 2
        bordered = cv2.copyMakeBorder(
            marker_img,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=(255.0,),
        )

        annotated = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)

        mx, my = border, border  # top-left of marker in bordered image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = pixel_size / 400
        thickness = max(1, int(pixel_size / 100))
        label_thick = max(1, thickness - 1)
        gap = border // 5  # padding between marker edge and labels

        # Corner labels in the border, outside the marker
        label_positions = [
            (mx - gap - int(font_scale * 10), my - gap),  # TL
            (mx + pixel_size + gap, my - gap),  # TR
            (mx + pixel_size + gap, my + pixel_size + gap + int(font_scale * 12)),  # BR
            (mx - gap - int(font_scale * 10), my + pixel_size + gap + int(font_scale * 12)),  # BL
        ]
        for i, (lx, ly) in enumerate(label_positions):
            cv2.putText(annotated, str(i), (lx, ly), font, font_scale, (0, 0, 0), thickness)

        # Axis legend: small square with arrows in the bottom-right border
        legend_size = border // 3
        legend_margin = border // 6
        # Bottom-right border area, below corner label 2
        lx = mx + pixel_size + legend_margin
        ly = my + pixel_size + border * 2 // 3

        # Draw a small gray square representing the marker
        cv2.rectangle(annotated, (lx, ly), (lx + legend_size, ly + legend_size), (200, 200, 200), -1)
        cv2.rectangle(annotated, (lx, ly), (lx + legend_size, ly + legend_size), (0, 0, 0), 1)

        # Origin dot at center of legend square
        cx = lx + legend_size // 2
        cy = ly + legend_size // 2
        cv2.circle(annotated, (cx, cy), max(2, thickness), (0, 0, 0), -1)

        arrow_len = legend_size // 2 + legend_margin
        arrow_thick = max(1, thickness)

        # X-axis: red, pointing right from center
        cv2.arrowedLine(annotated, (cx, cy), (cx + arrow_len, cy), (0, 0, 255), arrow_thick, tipLength=0.25)
        cv2.putText(
            annotated,
            "X",
            (cx + arrow_len + 2, cy + int(font_scale * 5)),
            font,
            font_scale * 0.4,
            (0, 0, 255),
            label_thick,
        )
        # Y-axis: green, pointing UP from center (negative y in image coords)
        cv2.arrowedLine(annotated, (cx, cy), (cx, cy - arrow_len), (0, 180, 0), arrow_thick, tipLength=0.25)
        cv2.putText(
            annotated,
            "Y",
            (cx - int(font_scale * 8), cy - arrow_len - int(font_scale * 3)),
            font,
            font_scale * 0.4,
            (0, 180, 0),
            label_thick,
        )

        # Info text at bottom of border
        size_cm = self.marker_size_m * 100
        info_y = my + pixel_size + border - int(font_scale * 5)
        cv2.putText(
            annotated,
            f"ID: {marker_id}  Size: {size_cm:.1f} cm",
            (mx, info_y),
            font,
            font_scale * 0.5,
            (0, 0, 0),
            label_thick,
        )

        return annotated
