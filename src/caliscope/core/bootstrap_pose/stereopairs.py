# caliscope/cameras/camera_array_initializer.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StereoPair:
    """
    Immutable representation of a stereo calibration result between two cameras.

    Encapsulates the extrinsic transformation from primary to secondary camera
    along with the calibration error score. The transformation answers:
    "Given a point in primary's coordinate frame, where is it in secondary's frame?"

    Translation is stored as a 1D array of shape (3,) for consistency across the codebase.
    This avoids shape-related broadcasting bugs in downstream operations.
    """

    primary_port: int
    secondary_port: int
    error_score: float
    translation: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        """Ensure translation is always shape (3,) and rotation is (3,3)."""
        # Squeeze translation to 1D if it's (3,1) or similar from OpenCV
        # Use object.__setattr__ because this is a frozen dataclass
        object.__setattr__(self, "translation", np.squeeze(self.translation))

        # Validate shapes for early error detection
        if self.translation.shape != (3,):
            raise ValueError(
                f"Translation must be shape (3,) after squeezing, got {self.translation.shape}. "
                "This usually indicates a bug in pose estimation or transformation composition."
            )
        if self.rotation.shape != (3, 3):
            raise ValueError(f"Rotation must be shape (3,3), got {self.rotation.shape}")

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.primary_port, self.secondary_port)

    @property
    def transformation(self) -> np.ndarray:
        """Get 4x4 transformation matrix [R|t; 0 0 0 1]."""
        R_stack = np.vstack([self.rotation, np.array([0, 0, 0])])
        t_stack = np.vstack([self.translation.reshape(3, 1), np.array([[1]])])
        return np.hstack([R_stack, t_stack])

    def inverted(self) -> StereoPair:
        """Reverse the link direction: A->B becomes B->A.

        Error score is preserved (same measurement, different direction).
        """
        inverted_transformation = np.linalg.inv(self.transformation)
        return StereoPair(
            primary_port=self.secondary_port,
            secondary_port=self.primary_port,
            error_score=self.error_score,
            rotation=inverted_transformation[0:3, 0:3],
            translation=inverted_transformation[0:3, 3],  # Single index gives (3,) shape
        )

    def link(self, other: StereoPair) -> StereoPair:
        """Extend this link through another: (A->B).link(B->C) = A->C.

        Error scores are summed as a conservative bound for the extended link.
        Caller is responsible for ensuring self.secondary_port == other.primary_port.
        """
        bridged_transformation = np.matmul(other.transformation, self.transformation)
        return StereoPair(
            primary_port=self.primary_port,
            secondary_port=other.secondary_port,
            error_score=self.error_score + other.error_score,
            rotation=bridged_transformation[0:3, 0:3],
            translation=bridged_transformation[0:3, 3],  # Single index gives (3,) shape
        )
