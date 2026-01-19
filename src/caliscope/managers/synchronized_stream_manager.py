import logging
import statistics
from pathlib import Path

import cv2

from caliscope.cameras.camera_array import CameraData
from caliscope.cameras.synchronizer import Synchronizer
from caliscope.tracker import Tracker
from caliscope.recording import FramePacketStreamer, create_streamer
from caliscope.recording.video_recorder import VideoRecorder
from caliscope.trackers.charuco_tracker import CharucoTracker

logger = logging.getLogger(__name__)


class SynchronizedStreamManager:
    """Orchestrates batch processing of multi-camera video to extract 2D landmarks.

    Takes a directory of concurrently recorded video and a Tracker, produces xy.csv
    output that is the foundation of extrinsic calibration and point triangulation.

    TODO(#890): Refactor to pure function `process_synchronized_recording()` with:
    - CancellationToken support for graceful shutdown
    - on_progress callback for TaskManager integration
    - on_sync_packet callback for live frame display
    See CLAUDE.md "Planned Refactor: SynchronizedStreamManager" for architecture.

    Current responsibilities:
    - Create FramePacketStreamer per camera
    - Create Synchronizer for frame alignment
    - Create VideoRecorder for output
    """

    def __init__(
        self,
        recording_dir: Path,
        all_camera_data: dict[int, CameraData],
        tracker: Tracker | CharucoTracker | None = None,
    ) -> None:
        self.recording_dir = recording_dir
        self.all_camera_data = all_camera_data
        self.tracker = tracker

        self.subfolder_name = "processed" if tracker is None else tracker.name
        self.output_dir = Path(self.recording_dir, self.subfolder_name)

        self.load_video_properties()

        # Initialized lazily in process_streams()
        self.streamers: dict[int, FramePacketStreamer] = {}
        self.synchronizer: Synchronizer | None = None
        self.recorder: VideoRecorder | None = None

    def process_streams(self, fps_target: int | None = None, include_video: bool = True) -> None:
        """
        Output file will be created in a subfolder named `tracker.name`
        This will include mp4 files with visualized landmarks as well as the file `xy.csv`
        Default behavior is to process streams at the mean frame rate they were recorded at.
        But this can be overridden with a new fps_target
        """
        if fps_target is None:
            fps_target = int(self.mean_fps)

        # Create streamers with fps_target
        self.streamers = {}
        for camera in self.all_camera_data.values():
            streamer = create_streamer(
                video_directory=self.recording_dir,
                port=camera.port,
                rotation_count=camera.rotation_count,
                tracker=self.tracker,
                fps_target=fps_target,
                end_behavior="stop",  # Stop at end for batch processing
            )
            self.streamers[camera.port] = streamer

        logger.info(f"Creating synchronizer based off of streamers: {self.streamers}")
        self.synchronizer = Synchronizer(self.streamers)
        self.synchronizer.start()  # Explicit start - Synchronizer no longer auto-starts in __init__
        self.recorder = VideoRecorder(self.synchronizer, suffix=self.subfolder_name)

        logger.info(f"beginning to create recording for files saved to {self.output_dir}")
        self.recorder.start_recording(
            self.output_dir,
            include_video=include_video,
            show_points=True,
            store_point_history=True,
        )

        logger.info(f"About to start playing video streamers: {self.streamers}")
        for port, streamer in self.streamers.items():
            streamer.start()

    def load_video_properties(self):
        fps = []
        frame_count = []
        logger.info(f"About to load video properties for files stored in {self.recording_dir}")
        logger.info(f"Current camera data is: {self.all_camera_data}")
        for camera in self.all_camera_data.values():
            mp4_path = Path(self.recording_dir, f"port_{camera.port}.mp4")

            video_properties = read_video_properties(mp4_path)
            fps.append(video_properties["fps"])
            frame_count.append(video_properties["frame_count"])
            logger.info(f"loading the following video properties: {video_properties}")

        self.mean_fps = statistics.mean(fps)
        self.mean_frame_count = statistics.mean(frame_count)

    def cleanup(self) -> None:
        """Stop all managed threads and release resources.

        Cleanup order matters for the pipeline:
        1. Recorder (downstream subscriber) - stop consuming sync packets
        2. Synchronizer (middle) - stop producing sync packets
        3. Streamers (upstream) - stop producing frame packets
        """
        logger.info("SynchronizedStreamManager cleanup initiated")

        # Stop recorder first (subscribes to synchronizer)
        if self.recorder is not None:
            self.recorder.stop_recording()
            logger.info("Recorder stopped")

        # Stop synchronizer (subscribes to streamers)
        if self.synchronizer is not None:
            self.synchronizer.stop()
            logger.info("Synchronizer stopped")

        # Close streamers last (close() calls stop() then cleans up tracker resources)
        for port, streamer in self.streamers.items():
            streamer.close()
            logger.info(f"Streamer for port {port} closed")

        logger.info("SynchronizedStreamManager cleanup complete")


def read_video_properties(source_path: Path) -> dict:
    # Dictionary to hold video properties
    properties = {}

    # Open the video file
    video = cv2.VideoCapture(str(source_path))
    logger.info(f"Attempting to open video file: {source_path}")

    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Could not open the video file: {source_path}")

    # Extract video properties
    properties["frame_count"] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    properties["fps"] = video.get(cv2.CAP_PROP_FPS)
    properties["width"] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties["height"] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    properties["size"] = (properties["width"], properties["height"])

    # Release the video capture object
    video.release()

    return properties
