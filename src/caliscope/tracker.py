from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from caliscope.packets import PixelFormat, PointPacket

logger = logging.getLogger(__name__)


class Tracker(ABC):
    @property
    def name(self) -> str:
        """
        returns the tracker name
        Used for file naming creation
        """
        return "Name Me"

    @property
    def pixel_format(self) -> PixelFormat:
        return PixelFormat.BGR

    def get_points(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        """Enforce pixel format contract, then delegate to _detect."""
        frame = self._ensure_format(frame)
        return self._detect(frame, cam_id, rotation_count)

    def _ensure_format(self, frame: np.ndarray) -> np.ndarray:
        if self.pixel_format == PixelFormat.GRAY and frame.ndim == 3:
            logger.warning(
                "%s received BGR frame, expected grayscale — converting. "
                "Pass pixel_format=tracker.pixel_format to FrameSource for zero-cost Y-plane extraction.",
                type(self).__name__,
            )
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.pixel_format == PixelFormat.BGR and frame.ndim == 2:
            logger.warning(
                "%s received grayscale frame, expected BGR — converting.",
                type(self).__name__,
            )
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    @abstractmethod
    def _detect(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        pass

    @abstractmethod
    def get_point_name(self, keypoint_id: int) -> str:
        """Maps keypoint_id to a name for data headers."""
        pass

    @abstractmethod
    def scatter_draw_instructions(self, keypoint_id: int) -> dict:
        """Maps keypoint_id to draw parameters (radius, color, thickness) for cv2.circle."""
        pass

    @property
    def wireframe(self) -> WireFrameView | None:
        """Wireframe topology for 3D visualization, or None if not applicable."""
        return None

    def get_connected_points(self) -> set[tuple[int, int]]:
        """
        OPTIONAL METHOD
        used for 2d drawing purposes elsewhere. Specify which
        points (if any) should have a line connecting them
        {(keypoint_id_A, keypoint_id_B),etc...}

        currently only implemented for charuco...
        """
        return set()

    def cleanup(self) -> None:
        """Release tracker resources (threads, GPU memory, etc.).

        Override in subclasses that spawn background threads or hold GPU resources.
        Default implementation is a no-op for stateless trackers (ArUco, Charuco).
        """
        pass


@dataclass(slots=True, frozen=True)
class Segment:
    name: str
    color: str  # one of: r, g, b, c, m, y, k, w
    point_A: str  # name of landmark
    point_B: str  # name of landmark
    width: float = 1  # note that this does not scale with zoom level... should probably just stick with 1


@dataclass(slots=True, frozen=True)
class WireFrameView:
    """Pure data container for wireframe visualization config.

    Stores segment definitions and point name→ID mapping.
    Rendering is handled by the visualization layer (PyVista widgets).
    """

    segments: tuple[Segment, ...]
    point_names: dict[str, int]  # map landmark name to landmark id
