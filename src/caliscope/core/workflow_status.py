"""Workflow status data structures for calibration progress tracking.

This module provides a frozen dataclass that captures a snapshot of calibration
workflow progress. The Coordinator computes this from ground truth (filesystem
and domain objects) and the Project tab displays it without duplicating logic.
"""

from dataclasses import dataclass
from enum import Enum, auto


class StepStatus(Enum):
    """Status of a workflow step."""

    NOT_STARTED = auto()  # Prerequisites not met
    INCOMPLETE = auto()  # In progress but not complete
    COMPLETE = auto()  # Fully complete
    AVAILABLE = auto()  # Optional feature, ready to use


@dataclass(frozen=True)
class WorkflowStatus:
    """Snapshot of calibration workflow progress.

    Computed by Coordinator.get_workflow_status() from ground truth.
    View uses this to render status display without duplicating logic.

    Fields are organized by workflow step:
    - Step 1: Project Setup (camera count, charuco)
    - Step 2: Intrinsic Calibration (per-camera intrinsics)
    - Step 3: Extrinsic 2D Extraction (synchronized landmark detection)
    - Step 4: Extrinsic Calibration (bundle adjustment for camera poses)
    - Step 5: Reconstruction (optional post-processing)
    """

    # Step 1: Project Setup
    camera_count: int
    charuco_configured: bool  # Always True after init

    # Step 2: Intrinsic Calibration
    intrinsic_videos_available: bool
    intrinsic_videos_missing: list[int]  # Ports with missing videos
    intrinsic_calibration_complete: bool
    cameras_needing_calibration: list[int]  # Ports without intrinsics

    # Step 3: Extrinsic 2D Extraction
    extrinsic_videos_available: bool
    extrinsic_videos_missing: list[int]  # Ports with missing videos
    extrinsic_2d_extraction_complete: bool

    # Step 4: Extrinsic Calibration
    extrinsic_calibration_complete: bool

    # Step 5: Reconstruction (optional)
    recordings_available: bool
    recording_names: list[str]

    @property
    def intrinsic_step_status(self) -> StepStatus:
        """Computed status for intrinsic calibration step."""
        if self.intrinsic_calibration_complete:
            return StepStatus.COMPLETE
        if self.intrinsic_videos_available:
            return StepStatus.INCOMPLETE
        return StepStatus.NOT_STARTED

    @property
    def extrinsic_2d_step_status(self) -> StepStatus:
        """Computed status for 2D extraction step."""
        if self.extrinsic_2d_extraction_complete:
            return StepStatus.COMPLETE
        if self.extrinsic_videos_available and self.intrinsic_calibration_complete:
            return StepStatus.INCOMPLETE
        return StepStatus.NOT_STARTED

    @property
    def extrinsic_calibration_step_status(self) -> StepStatus:
        """Computed status for extrinsic calibration step."""
        if self.extrinsic_calibration_complete:
            return StepStatus.COMPLETE
        if self.extrinsic_2d_extraction_complete:
            return StepStatus.INCOMPLETE
        return StepStatus.NOT_STARTED
