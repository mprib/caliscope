"""Frame timestamp mapping for synchronized video playback.

FrameTimestamps maps frame indices to wall-clock timestamps recorded at capture time.
This enables synchronized playback across multiple cameras.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Self

import pandas as pd


@dataclass(frozen=True, slots=True)
class FrameTimestamps:
    """Maps frame indices to timestamps recorded at capture time.

    Frame indices may not start at 0 for synchronized recordings where
    cameras started at different times.

    Attributes:
        frame_times: Immutable mapping of frame_index -> timestamp (seconds).
    """

    frame_times: Mapping[int, float]

    @property
    def start_frame_index(self) -> int:
        """First valid frame index (may not be 0 for synced recordings)."""
        return min(self.frame_times.keys())

    @property
    def last_frame_index(self) -> int:
        """Last valid frame index."""
        return max(self.frame_times.keys())

    def get_time(self, frame_index: int) -> float:
        """Get wall-clock timestamp for a frame index.

        Raises:
            KeyError: If frame_index is not in the mapping.
        """
        return self.frame_times[frame_index]

    @classmethod
    def from_csv(cls, csv_path: Path, port: int) -> Self:
        """Load timing from frame_time_history.csv.

        Frame indices are computed via rank-ordering of frame_time within
        the port's rows. This handles synchronized recordings where frames
        may not start at index 0, and ensures sequential indices even if
        there are gaps in the recorded timestamps.

        Args:
            csv_path: Path to frame_time_history.csv.
            port: Camera port to extract timing for.

        Raises:
            FileNotFoundError: If csv_path doesn't exist.
            KeyError: If port not found in CSV.
        """
        df = pd.read_csv(csv_path)
        port_df = df[df["port"] == port].copy()

        if port_df.empty:
            raise KeyError(f"Port {port} not found in {csv_path}")

        # Rank-based indexing: ensures sequential indices from timestamps
        port_df["frame_index"] = port_df["frame_time"].rank(method="min").astype(int) - 1

        frame_times = dict(zip(port_df["frame_index"], port_df["frame_time"]))
        return cls(MappingProxyType(frame_times))

    @classmethod
    def inferred(cls, fps: float, frame_count: int) -> Self:
        """Create timing inferred from FPS when no CSV exists.

        Generates timestamps assuming constant frame rate starting at t=0.

        Args:
            fps: Frames per second.
            frame_count: Total number of frames.
        """
        frame_times = {i: i / fps for i in range(frame_count)}
        return cls(MappingProxyType(frame_times))
