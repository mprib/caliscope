"""Recording module - video I/O, timing, and streaming.

Qt-tainted symbols (FramePacketStreamer, create_streamer) are NOT re-exported here.
Import them directly from caliscope.recording.frame_packet_streamer when needed.
"""

from caliscope.recording.frame_source import FrameSource
from caliscope.recording.frame_timestamps import FrameTimestamps
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
from caliscope.recording.video_utils import VideoProperties, read_video_properties

__all__ = [
    "FrameSource",
    "FrameTimestamps",
    "SynchronizedTimestamps",
    "VideoProperties",
    "read_video_properties",
]
