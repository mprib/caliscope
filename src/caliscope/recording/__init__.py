"""Recording module - video I/O, timing, and streaming."""

from caliscope.recording.frame_packet_streamer import (
    FramePacketStreamer,
    create_streamer,
)
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.frame_sync import SyncMap, compute_sync_indices
from caliscope.recording.frame_timestamps import FrameTimestamps
from caliscope.recording.video_utils import VideoProperties, read_video_properties

__all__ = [
    "FramePacketStreamer",
    "FrameSource",
    "FrameTimestamps",
    "SyncMap",
    "VideoProperties",
    "compute_sync_indices",
    "create_streamer",
    "read_video_properties",
]
