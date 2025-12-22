"""
Wireframe loading and geometry building - pure UI layer.
Zero domain dependencies beyond Tracker interface.
"""

from dataclasses import dataclass
from pathlib import Path

import rtoml


@dataclass(frozen=True)
class WireframeSegment:
    """Runtime wireframe segment with resolved point IDs."""

    name: str
    point_a_id: int
    point_b_id: int
    color_rgb: tuple[float, float, float]


def load_wireframe_segments(toml_path: Path, point_name_to_id: dict[str, int]) -> list[WireframeSegment]:
    """
    Load wireframe from TOML and resolve point names to IDs.

    Args:
        toml_path: Path to wireframe definition file
        point_name_to_id: Mapping from point names to integer IDs

    Returns:
        List of wireframe segments ready for geometry building
    """
    toml_data = rtoml.load(toml_path)
    segments: list[WireframeSegment] = []

    for segment_name, specs in toml_data.items():
        point_names: list[str] = specs["points"]

        if len(point_names) != 2:
            raise ValueError(f"Segment {segment_name} must have exactly 2 points")

        # Resolve names to IDs immediately
        point_a_id = point_name_to_id[point_names[0]]
        point_b_id = point_name_to_id[point_names[1]]

        # Get color with default
        color: list[float] = specs.get("color", [0.5, 0.5, 0.5])
        if len(color) != 3:
            raise ValueError(f"Color for {segment_name} must be 3 values [R,G,B]")

        segments.append(
            WireframeSegment(
                name=segment_name,
                point_a_id=point_a_id,
                point_b_id=point_b_id,
                color_rgb=tuple(color),
            )
        )

    return segments
