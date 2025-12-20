# caliscope/cameras/camera_array_initializer.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class StereoPair:
    """
    A dataclass to hold the extrinsic parameters associated with the cv2.stereoCalibrate
    function output. Additionally provides some convenience methods to get common transformations
    of the data.

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
        self.translation = np.squeeze(self.translation)

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


def get_inverted_stereopair(stereo_pair: StereoPair) -> StereoPair:
    """
    Create a StereoPair with inverted transformation (secondary->primary).

    The inversion operation preserves the error_score and creates a new StereoPair
    with primary/secondary ports swapped and the transformation inverted.
    """
    inverted_transformation = np.linalg.inv(stereo_pair.transformation)
    return StereoPair(
        primary_port=stereo_pair.secondary_port,
        secondary_port=stereo_pair.primary_port,
        error_score=stereo_pair.error_score,
        rotation=inverted_transformation[0:3, 0:3],
        translation=inverted_transformation[0:3, 3],  # Use single index to get (3,) shape
    )


def get_bridged_stereopair(pair_A_B: StereoPair, pair_B_C: StereoPair) -> StereoPair:
    """
    Create a StereoPair by chaining A->B and B->C transformations.

    The error_score is summed, and the transformation is the matrix product
    of the two transformations (B_C @ A_B).
    """
    port_A = pair_A_B.primary_port
    port_C = pair_B_C.secondary_port
    A_C_error = pair_A_B.error_score + pair_B_C.error_score

    bridged_transformation = np.matmul(pair_B_C.transformation, pair_A_B.transformation)
    return StereoPair(
        primary_port=port_A,
        secondary_port=port_C,
        error_score=A_C_error,
        rotation=bridged_transformation[0:3, 0:3],
        translation=bridged_transformation[0:3, 3],  # Use single index to get (3,) shape
    )
