"""Pixel-dimension compatibility checks for one recording session."""

from dataclasses import dataclass
from pathlib import Path

from caliscope.recording.video_utils import read_video_dimensions


@dataclass(frozen=True)
class CameraDimensionOutcome:
    """The recorded and calibrated dimensions for one required camera."""

    cam_id: int
    expected_size: tuple[int, int] | None
    actual_size: tuple[int, int] | None
    error: str | None


@dataclass(frozen=True)
class RecordingDimensionAssessment:
    """Immutable dimension-check result for a single recording session."""

    outcomes: tuple[CameraDimensionOutcome, ...]

    @property
    def is_compatible(self) -> bool:
        """Whether every supplied camera has the exact calibrated dimensions."""
        return bool(self.outcomes) and all(
            outcome.error is None and outcome.expected_size is not None and outcome.actual_size == outcome.expected_size
            for outcome in self.outcomes
        )


def check_recording_dimensions(
    recording_dir: Path,
    expected_sizes: tuple[tuple[int, tuple[int, int] | None], ...],
) -> RecordingDimensionAssessment:
    """Check supplied recording cameras against their calibrated dimensions.

    This intentionally checks only the supplied camera IDs.  Structural video
    naming and membership remain the WorkspaceGuide's responsibility.
    """
    outcomes: list[CameraDimensionOutcome] = []
    for cam_id, expected_size in sorted(expected_sizes, key=lambda item: item[0]):
        if not _is_valid_size(expected_size):
            outcomes.append(
                CameraDimensionOutcome(
                    cam_id=cam_id,
                    expected_size=expected_size,
                    actual_size=None,
                    error=f"Calibration dimensions are missing or invalid for camera {cam_id}.",
                )
            )
            continue

        video_path = recording_dir / f"cam_{cam_id}.mp4"
        try:
            actual_size = read_video_dimensions(video_path)
        except (FileNotFoundError, ValueError) as error:
            outcomes.append(
                CameraDimensionOutcome(
                    cam_id=cam_id,
                    expected_size=expected_size,
                    actual_size=None,
                    error=f"Could not read video dimensions for {video_path}: {error}",
                )
            )
            continue

        outcomes.append(
            CameraDimensionOutcome(
                cam_id=cam_id,
                expected_size=expected_size,
                actual_size=actual_size,
                error=None,
            )
        )

    return RecordingDimensionAssessment(outcomes=tuple(outcomes))


def _is_valid_size(size: tuple[int, int] | None) -> bool:
    return (
        size is not None
        and len(size) == 2
        and all(isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0 for dimension in size)
    )
