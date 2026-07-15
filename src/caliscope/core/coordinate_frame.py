"""Building a canonical Z-up world basis from a gravity vector and a heading.

Ported from monokin's ``coordinate_frame.py``. The numerics are identical:
up becomes +Z, forward's horizontal component becomes +Y (the yaw anchor),
and +X completes the right-handed basis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def world_basis_from_up_and_forward(up: NDArray, *, forward: NDArray) -> NDArray:
    """(3,3) rotation into the Z-up world frame: rows are the world basis vectors.

    up (gravity-up, normalized here) becomes world +Z; forward's horizontal
    component becomes world +Y, the yaw anchor, so the source frame's forward
    direction unfolds along +Y; +X completes the right-handed basis. Apply as
    p_world = R @ p_source. Raises when forward is within ~1e-6 of vertical,
    where the yaw anchor is undefined.
    """
    z_world = np.asarray(up, dtype=np.float64)
    z_world = z_world / np.linalg.norm(z_world)
    horizontal = np.asarray(forward, dtype=np.float64)
    horizontal = horizontal - np.dot(horizontal, z_world) * z_world
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-6:
        raise ValueError("forward points along gravity (pitch ~ +/-90 deg); the forward-to-+Y yaw anchor is undefined")
    y_world = horizontal / norm
    x_world = np.cross(y_world, z_world)
    return np.stack([x_world, y_world, z_world])
