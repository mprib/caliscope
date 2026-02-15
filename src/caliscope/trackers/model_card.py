import logging
from dataclasses import dataclass
from pathlib import Path

import rtoml

from caliscope.tracker import Segment, WireFrameView

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Configuration for an ONNX tracking model.

    Captures model metadata, input parameters, output format, and visualization
    config (point names and wireframe). Supports two output formats: SimCC
    (coordinate classification) and heatmap-based models.
    """

    name: str
    model_path: Path
    format: str  # "simcc" or "heatmap"
    input_width: int
    input_height: int
    confidence_threshold: float
    point_name_to_id: dict[str, int]
    wireframe: WireFrameView | None  # None if no segments defined

    @property
    def point_id_to_name(self) -> dict[int, str]:
        """Reverse mapping from point ID to landmark name."""
        return {v: k for k, v in self.point_name_to_id.items()}

    @property
    def onnx_exists(self) -> bool:
        """Check if the ONNX model file is present on disk.

        Useful for GUI status display — model card can be loaded even if
        the .onnx file hasn't been downloaded yet.
        """
        return self.model_path.exists()

    @staticmethod
    def from_toml(path: Path) -> "ModelCard":
        """Load and validate a model card TOML file.

        Does NOT validate that model_path exists (use onnx_exists property for that).
        Does validate all required TOML fields are present and correctly typed.

        Raises:
            FileNotFoundError: If the TOML file itself doesn't exist
            ValueError: If required fields are missing or malformed
        """
        if not path.exists():
            raise FileNotFoundError(f"Model card TOML not found: {path}")

        config = rtoml.load(path)

        # Validate [model] section exists
        if "model" not in config:
            raise ValueError(f"Missing required [model] section in {path}")

        model_section = config["model"]

        # Validate required fields
        if "model_path" not in model_section:
            raise ValueError(f"Missing required field 'model_path' in [model] section of {path}")

        if "format" not in model_section:
            raise ValueError(f"Missing required field 'format' in [model] section of {path}")

        if "input_size" not in model_section:
            raise ValueError(f"Missing required field 'input_size' in [model] section of {path}")

        # Validate format
        format_value = model_section["format"]
        if format_value not in ("simcc", "heatmap"):
            raise ValueError(f"Invalid format '{format_value}' in {path}. Must be 'simcc' or 'heatmap'.")

        # Validate input_size
        input_size = model_section["input_size"]
        if not isinstance(input_size, list) or len(input_size) != 2:
            raise ValueError(f"input_size must be a 2-element list [width, height], got {input_size} in {path}")

        # Parse [points] section
        if "points" not in config:
            raise ValueError(f"Missing required [points] section in {path}")

        point_name_to_id = config["points"]
        if not isinstance(point_name_to_id, dict):
            raise ValueError(f"[points] section must be a dictionary, got {type(point_name_to_id)} in {path}")

        # Parse [segments.*] sections into wireframe
        # TOML parses [segments.shoulders] as config["segments"]["shoulders"]
        segments: list[Segment] = []
        segments_section = config.get("segments", {})
        for segment_name, segment_data in segments_section.items():
            if "color" not in segment_data:
                raise ValueError(f"Segment [segments.{segment_name}] missing 'color' field in {path}")
            if "points" not in segment_data:
                raise ValueError(f"Segment [segments.{segment_name}] missing 'points' field in {path}")

            points = segment_data["points"]
            if not isinstance(points, list) or len(points) != 2:
                raise ValueError(
                    f"Segment [segments.{segment_name}] 'points' must be a 2-element list, got {points} in {path}"
                )

            segment = Segment(
                name=segment_name,
                color=segment_data["color"],
                point_A=points[0],
                point_B=points[1],
                width=segment_data.get("width", 1.0),
            )
            segments.append(segment)

        wireframe = WireFrameView(segments=tuple(segments), point_names=point_name_to_id) if segments else None

        # Construct ModelCard
        return ModelCard(
            name=model_section.get("name", Path(model_section["model_path"]).stem),
            model_path=Path(model_section["model_path"]),
            format=format_value,
            input_width=input_size[0],
            input_height=input_size[1],
            confidence_threshold=model_section.get("confidence_threshold", 0.3),
            point_name_to_id=point_name_to_id,
            wireframe=wireframe,
        )
