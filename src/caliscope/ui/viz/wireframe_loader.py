"""
Wireframe loading and geometry building - pure UI layer.
Zero domain dependencies beyond Tracker interface.
"""

from dataclasses import dataclass
from pathlib import Path

import rtoml


# Map PyQtGraph single-character colors to RGB
COLOR_MAP = {
    "r": (1.0, 0.0, 0.0),
    "g": (0.0, 1.0, 0.0),
    "b": (0.0, 0.0, 1.0),
    "c": (0.0, 1.0, 1.0),
    "m": (1.0, 0.0, 1.0),
    "y": (1.0, 1.0, 0.0),
    "k": (0.0, 0.0, 0.0),
    "w": (1.0, 1.0, 1.0),
}


@dataclass(frozen=True)
class WireframeSegment:
    """Runtime wireframe segment with resolved point IDs."""

    name: str
    point_a_id: int
    point_b_id: int
    color_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class WireframeConfig:
    """Complete wireframe configuration with point mapping."""

    point_name_to_id: dict[str, int]
    segments: list[WireframeSegment]


def load_wireframe_config(toml_path: Path) -> WireframeConfig:
    """
    Load complete wireframe configuration from TOML.

    Args:
        toml_path: Path to wireframe definition file

    Returns:
        WireframeConfig with point mapping and segments
    """
    toml_data = rtoml.load(toml_path)

    # Load point name to ID mapping
    point_name_to_id = toml_data.get("points", {})
    if not point_name_to_id:
        raise ValueError("TOML must contain [points] section with name-to-ID mapping")

    # Load segments
    segments: list[WireframeSegment] = []
    for segment_name, specs in toml_data.items():
        if segment_name == "points":
            continue  # Skip the points mapping section

        point_names: list[str] = specs["points"]
        if len(point_names) != 2:
            raise ValueError(f"Segment {segment_name} must have exactly 2 points")

        # Resolve names to IDs using the mapping from the same file
        if point_names[0] not in point_name_to_id or point_names[1] not in point_name_to_id:
            continue  # Skip segments with missing points

        point_a_id = point_name_to_id[point_names[0]]
        point_b_id = point_name_to_id[point_names[1]]

        # Get color and convert to RGB
        color_code: str = specs.get("color", "w")
        if color_code not in COLOR_MAP:
            raise ValueError(f"Invalid color code '{color_code}' in segment {segment_name}")

        segments.append(
            WireframeSegment(
                name=segment_name,
                point_a_id=point_a_id,
                point_b_id=point_b_id,
                color_rgb=COLOR_MAP[color_code],
            )
        )

    return WireframeConfig(point_name_to_id=point_name_to_id, segments=segments)
