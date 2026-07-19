"""Metric scale cues for anchoring a shape-only capture volume.

Bundle adjustment determines shape up to a similarity transform. Each cue
supplies one observation of true metric distance, compiled by
``CaptureVolume.scaled()`` to an (arbitrary-units distance, metric meters,
sigma) triple. One cue scales exactly; multiple cues combine via
sigma-weighted least squares.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDistance:
    """Known distance between two camera centers (e.g., tape measure)."""

    cam_a: int  # cam_id
    cam_b: int  # cam_id
    meters: float
    sigma_m: float = 0.01  # tape measure accuracy


@dataclass(frozen=True)
class SegmentLength:
    """Known distance between two tracked keypoints (e.g., a body segment)."""

    keypoint_id_a: int  # keypoint id from ImagePoints
    keypoint_id_b: int  # keypoint id from ImagePoints
    meters: float
    sigma_m: float = 0.02


@dataclass(frozen=True)
class DepthObservation:
    """Metric depth of one keypoint in one camera at one moment."""

    cam_id: int
    keypoint_id: int
    sync_index: int
    depth_m: float
    sigma_m: float = 0.1  # monocular depth noise
