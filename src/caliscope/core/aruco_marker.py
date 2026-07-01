from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import rtoml
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArucoMarker:
    marker_id: int
    size_m: float

    def __post_init__(self) -> None:
        if self.size_m <= 0:
            raise ValueError(f"size_m must be positive, got {self.size_m}")

    @cached_property
    def corners(self) -> NDArray[np.float64]:
        """(4, 3) corner positions in marker-local frame.
        Origin at center, X right, Y up, Z=0. Ordered TL, TR, BR, BL."""
        s = self.size_m / 2
        return np.array(
            [[-s, +s, 0.0], [+s, +s, 0.0], [+s, -s, 0.0], [-s, -s, 0.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ArucoMarkerSet:
    dictionary: int
    markers: dict[int, ArucoMarker]

    def __post_init__(self) -> None:
        if not self.markers:
            raise ValueError("ArucoMarkerSet requires at least one marker")
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        capacity = len(aruco_dict.bytesList)
        for mid in self.markers:
            if mid < 0 or mid >= capacity:
                raise ValueError(f"Marker ID {mid} exceeds dictionary capacity ({capacity})")

    @classmethod
    def from_toml(cls, path: Path) -> ArucoMarkerSet:
        from caliscope.persistence import PersistenceError

        if not path.exists():
            raise PersistenceError(f"ArucoMarkerSet file not found: {path}")
        try:
            data = rtoml.load(path)
            dictionary = data["dictionary"]
            markers = {}
            for entry in data.get("markers", []):
                mid = entry["id"]
                size_m = entry["size_m"]
                markers[mid] = ArucoMarker(marker_id=mid, size_m=size_m)
            return cls(dictionary=dictionary, markers=markers)
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to load ArucoMarkerSet from {path}: {e}") from e

    def to_toml(self, path: Path) -> None:
        from caliscope.persistence import PersistenceError, _safe_write_toml

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "dictionary": self.dictionary,
                "markers": [
                    {"id": m.marker_id, "size_m": m.size_m}
                    for m in sorted(self.markers.values(), key=lambda m: m.marker_id)
                ],
            }
            _safe_write_toml(data, path)
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to save ArucoMarkerSet to {path}: {e}") from e

    def generate_marker_image(self, marker_id: int, pixel_size: int) -> NDArray:
        if marker_id not in self.markers:
            raise KeyError(f"Marker {marker_id} not in set (available: {sorted(self.markers.keys())})")
        marker = self.markers[marker_id]
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

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = pixel_size / 400
        thickness = max(1, int(pixel_size / 100))
        label_thick = max(1, thickness - 1)

        size_cm = marker.size_m * 100
        info_y = border + pixel_size + border - int(font_scale * 5)
        cv2.putText(
            annotated,
            f"ID: {marker_id}  Size: {size_cm:.1f} cm",
            (border, info_y),
            font,
            font_scale * 0.5,
            (0, 0, 0),
            label_thick,
        )
        return annotated
