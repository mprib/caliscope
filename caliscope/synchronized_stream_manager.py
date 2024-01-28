import caliscope.logger
import statistics
import cv2
from pathlib import Path
from caliscope.cameras.synchronizer import Synchronizer
from caliscope.recording.recorded_stream import RecordedStream
from caliscope.cameras.camera_array import CameraData
from caliscope.packets import Tracker
from caliscope.recording.video_recorder import VideoRecorder

logger = caliscope.logger.get(__name__)


class SynchronizedStreamManager:
    """
    The primary job of the SynchronizedStreamManager is to take in a directory of concurrently recorded video
    as well as a Tracker and produce the xy.csv file that is the foundation of both the extrinsic calibration
    as well as the point triangulation.

    Related to this, it also directs where the recorded data will go and builds the DictionaryFrameEmitter
    that will broadcast a dictionary of frames while they are being processed.

    Because of this it has substantial and broad responsibilities that include creation of:
    - streams
    - synchronizer
    - video recorder
    """

    def __init__(
        self,
        recording_dir: Path,
        all_camera_data: dict[CameraData],
        tracker: Tracker = None,
    ) -> None:
        self.recording_dir = recording_dir
        self.all_camera_data = all_camera_data
        self.tracker = tracker

        self.subfolder_name = "processed" if tracker is None else self.tracker.name
        self.output_dir = Path(self.recording_dir, self.subfolder_name)

        self.load_video_properties()
        # To be filled when loading stream tools
        self.load_stream_tools()

    def load_stream_tools(self):
        self.streams = {}

        for camera in self.all_camera_data.values():
            stream = RecordedStream(
                directory=self.recording_dir,
                port=camera.port,
                rotation_count=camera.rotation_count,
                tracker=self.tracker,
                break_on_last=True,
            )

            self.streams[camera.port] = stream

        logger.info(f"Creating synchronizer based off of streams: {self.streams}")
        self.synchronizer = Synchronizer(self.streams)
        self.recorder = VideoRecorder(self.synchronizer, suffix=self.subfolder_name)

    def process_streams(self, fps_target=None):
        """
        Output file will be created in a subfolder named `tracker.name`
        This will include mp4 files with visualized landmarks as well as the file `xy.csv`
        Default behavior is to process streams at the mean frame rate they were recorded at.
        But this can be overridden with a new fps_target
        """
        logger.info(f"beginning to create recording for files saved to {self.output_dir}")
        self.recorder.start_recording(
            self.output_dir,
            include_video=True,
            show_points=True,
            store_point_history=True,
        )

        if fps_target is None:
            fps_target = self.mean_fps
        
        logger.info(f"About to start playing video streams to be processed. Streams: {self.streams}")
        for port, stream in self.streams.items():

            if fps_target is not None:
                stream.set_fps_target(fps_target)

            stream.play_video()
            
    def load_video_properties(self):
        fps   = []
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