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
    """

    primary_port: int
    secondary_port: int
    error_score: float
    translation: np.ndarray
    rotation: np.ndarray

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.primary_port, self.secondary_port)

    @property
    def transformation(self) -> np.ndarray:
        R_stack = np.vstack([self.rotation, np.array([0, 0, 0])])
        t_stack = np.vstack([self.translation.reshape(3, 1), np.array([[1]])])
        return np.hstack([R_stack, t_stack])


def get_inverted_stereopair(stereo_pair: StereoPair) -> StereoPair:
    inverted_transformation = np.linalg.inv(stereo_pair.transformation)
    return StereoPair(
        primary_port=stereo_pair.secondary_port,
        secondary_port=stereo_pair.primary_port,
        error_score=stereo_pair.error_score,
        rotation=inverted_transformation[0:3, 0:3],
        translation=inverted_transformation[0:3, 3:],
    )


def get_bridged_stereopair(pair_A_B: StereoPair, pair_B_C: StereoPair) -> StereoPair:
    port_A = pair_A_B.primary_port
    port_C = pair_B_C.secondary_port
    A_C_error = pair_A_B.error_score + pair_B_C.error_score

    bridged_transformation = np.matmul(pair_B_C.transformation, pair_A_B.transformation)
    return StereoPair(
        primary_port=port_A,
        secondary_port=port_C,
        error_score=A_C_error,
        rotation=bridged_transformation[0:3, 0:3],
        translation=bridged_transformation[0:3, 3:],
    )
