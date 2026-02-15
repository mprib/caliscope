"""
Wireframe loading and geometry building - UI layer.
Depends on domain WireFrameView for conversion to GUI-ready segments.
"""

from dataclasses import dataclass
from pathlib import Path

import rtoml

from caliscope.tracker import WireFrameView


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


def wireframe_segments_from_view(view: WireFrameView) -> list[WireframeSegment]:
    """Convert domain WireFrameView to GUI-ready WireframeSegment list.

    Resolves landmark names to IDs and single-char color codes to RGB tuples.
    Silently skips segments with unresolved point names or unknown color codes.
    """
    segments: list[WireframeSegment] = []
    for seg in view.segments:
        point_a_id = view.point_names.get(seg.point_A)
        point_b_id = view.point_names.get(seg.point_B)
        color_rgb = COLOR_MAP.get(seg.color)

        if point_a_id is None or point_b_id is None or color_rgb is None:
            continue

        segments.append(
            WireframeSegment(
                name=seg.name,
                point_a_id=point_a_id,
                point_b_id=point_b_id,
                color_rgb=color_rgb,
            )
        )
    return segments
