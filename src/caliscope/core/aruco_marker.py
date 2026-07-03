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
    static: bool = False

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
class MarkerLink:
    """Two markers on the same physical object with known corner correspondence."""

    marker_a: int
    marker_b: int
    corner_map: tuple[int, int, int, int]
    separation_m: float = 0.0

    def __post_init__(self) -> None:
        if sorted(self.corner_map) != [0, 1, 2, 3]:
            raise ValueError(f"corner_map must be a permutation of (0,1,2,3), got {self.corner_map}")
        if self.separation_m < 0:
            raise ValueError(f"separation_m must be >= 0, got {self.separation_m}")


@dataclass(frozen=True)
class ArucoMarkerSet:
    dictionary: int
    markers: dict[int, ArucoMarker]
    links: tuple[MarkerLink, ...] = ()

    def __post_init__(self) -> None:
        if not self.markers:
            raise ValueError("ArucoMarkerSet requires at least one marker")
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        capacity = len(aruco_dict.bytesList)
        for mid, marker in self.markers.items():
            if marker.marker_id != mid:
                raise ValueError(f"Key {mid} does not match marker_id {marker.marker_id}")
            if mid < 0 or mid >= capacity:
                raise ValueError(f"Marker ID {mid} exceeds dictionary capacity ({capacity})")
        for link in self.links:
            if link.marker_a not in self.markers:
                raise ValueError(f"MarkerLink references unknown marker_a={link.marker_a}")
            if link.marker_b not in self.markers:
                raise ValueError(f"MarkerLink references unknown marker_b={link.marker_b}")

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
                markers[mid] = ArucoMarker(marker_id=mid, size_m=size_m, static=entry.get("static", False))
            links = []
            for entry in data.get("links", []):
                links.append(
                    MarkerLink(
                        marker_a=entry["marker_a"],
                        marker_b=entry["marker_b"],
                        corner_map=tuple(entry["corner_map"]),
                        separation_m=entry.get("separation_m", 0.0),
                    )
                )
            return cls(dictionary=dictionary, markers=markers, links=tuple(links))
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to load ArucoMarkerSet from {path}: {e}") from e

    def to_toml(self, path: Path) -> None:
        from caliscope.persistence import PersistenceError, _safe_write_toml

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            markers_data = []
            for m in sorted(self.markers.values(), key=lambda m: m.marker_id):
                entry = {"id": m.marker_id, "size_m": m.size_m}
                if m.static:
                    entry["static"] = True
                markers_data.append(entry)
            data: dict = {
                "dictionary": self.dictionary,
                "markers": markers_data,
            }
            if self.links:
                data["links"] = [
                    {
                        "marker_a": link.marker_a,
                        "marker_b": link.marker_b,
                        "corner_map": list(link.corner_map),
                        "separation_m": link.separation_m,
                    }
                    for link in self.links
                ]
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

        corner_scale = font_scale * 0.4
        pad = int(font_scale * 12)
        corner_positions = [
            (border - pad, border - pad // 2),
            (border + pixel_size + pad // 4, border - pad // 2),
            (border + pixel_size + pad // 4, border + pixel_size + pad),
            (border - pad, border + pixel_size + pad),
        ]
        for idx, pos in enumerate(corner_positions):
            cv2.putText(annotated, str(idx), pos, font, corner_scale, (0, 0, 0), label_thick)

        return annotated
