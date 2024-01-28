import caliscope.logger
from time import sleep
from pathlib import Path
from caliscope.recording.recorded_stream import RecordedStream
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.cameras.camera_array import CameraData
from caliscope.packets import Tracker
from caliscope.gui.frame_emitters.playback_frame_emitter import PlaybackFrameEmitter
from caliscope.calibration.intrinsic_calibrator import IntrinsicCalibrator

logger = caliscope.logger.get(__name__)


class IntrinsicStreamManager:
    def __init__(
        self,
        recording_dir: Path,
        cameras: dict[CameraData],
        tracker: Tracker = None,
    ) -> None:
        self.recording_dir = recording_dir
        self.cameras = cameras
        self.tracker = tracker
        self.load_stream_tools()

    def load_stream_tools(self):
        self.streams = {}
        self.calibrators = {}
        self.frame_emitters = {}

        for camera in self.cameras.values():
            logger.info(f"Loading stream tools associated with camera {camera.port}")
            stream = RecordedStream(
                directory=self.recording_dir,
                port=camera.port,
                rotation_count=camera.rotation_count,
                tracker=self.tracker,
                break_on_last=False,
            )

            self.streams[camera.port] = stream

            # note: must create calibrator first so that grid history q can
            # then be passed to the frame_emitter
            self.calibrators[camera.port] = IntrinsicCalibrator(camera, stream)
            self.frame_emitters[camera.port] = PlaybackFrameEmitter(
                stream, grid_history_q=self.calibrators[camera.port].grid_history_q
            )
            self.frame_emitters[camera.port].start()

    def close_stream_tools(self):
        for port, emitter in self.frame_emitters.items():
            logger.info(f"Beginning to shut down frame emitter for port {port}")
            emitter.stop()

            logger.info(f"Waiting on camera {port} emitter to wrap up...")
            emitter.wait()
            logger.info(f"Finished waiting for camera {port} emitter to wrap up")

        for port, calibrator in self.calibrators.items():
            logger.info("stopping calibrator")
            calibrator.stop()

        for port, stream in self.streams.items():
            stream.stop_event.set()
            stream.unpause()
            logger.info(f"About to wait for camera {port} to close")
            # stream.jump_to(0)
            stream.thread.join()

        logger.info("Finished closing stream tools")

    def get_frame_count(self, port):
        """
        Note that if frame_index may not start at 0 if frame_history is from a real-time capture
        """
        start_frame_index = self.streams[port].start_frame_index
        last_frame_index = self.streams[port].last_frame_index

        return last_frame_index - start_frame_index + 1

    def update_charuco(self, charuco_tracker: CharucoTracker):
        for stream in self.streams.values():
            stream.tracker = charuco_tracker

        for emitter in self.frame_emitters.values():
            emitter.initialize_grid_capture_history()

    def play_stream(self, port):
        logger.info(f"Begin playing stream at port {port}")
        self.streams[port].play_video()

    def pause_stream(self, port):
        logger.info(f"Pausing stream at port {port}")
        self.streams[port].pause()

    def unpause_stream(self, port):
        logger.info(f"Unpausing stream at port {port}")
        self.streams[port].unpause()

    def stream_jump_to(self, port, frame):
        self.streams[port].jump_to(frame)

    def end_stream(self, port):
        self.streams[port].stop_event.set()
        self.unpause_stream(port)

    def add_calibration_grid(self, port: int, frame_index: int):
        self.calibrators[port].add_calibration_frame_index(frame_index)

    def clear_calibration_data(self, port: int):
        self.calibrators[port].clear_calibration_data()
        self.frame_emitters[port].initialize_grid_capture_history()

    def calibrate_camera(self, port: int):
        logger.info(f"Calibrating camera at port {port}")
        self.calibrators[port].calibrate_camera()

    def apply_distortion(self, camera: CameraData, undistort: bool):
        self.frame_emitters[camera.port].update_distortion_params(
            undistort=undistort, matrix=camera.matrix, distortions=camera.distortions
        )

    def set_stream_rotation(self, port, rotation_count):
        self.streams[port].rotation_count = rotation_count

    def autocalibrate(self, port, grid_count, pct_board_threshold):
        stream = self.streams[port]
        intrinsic_calibrator = self.calibrators[port]
        frame_emitter = self.frame_emitters[port]

        board_corners = self.tracker.charuco.board.getChessboardCorners()
        total_corner_count = board_corners.shape[0]
        threshold_corner_count = total_corner_count * pct_board_threshold
        threshold_corner_count = max(threshold_corner_count,6)   # additional requirement that I believe is part of the alogrithm

        logger.info(f"Corners for charuco are {board_corners}")

        # calculate basic wait time between board collections 
        # if many frames have incomplete data, this will fail to reach the target board count
        start_frame_index = stream.start_frame_index
        last_frame_index = stream.last_frame_index
        total_frames = last_frame_index - start_frame_index + 1
        wait_between = int(total_frames / grid_count)

        stream.set_fps_target(100)  # speed through the stream

        # jump to first frame, play videos and cycle quickly through frames
        stream.jump_to(0)
        frame_emitter.initialize_grid_capture_history()
        intrinsic_calibrator.initiate_auto_pop(
            wait_between=wait_between,
            threshold_corner_count=threshold_corner_count,
            target_grid_count=grid_count,
        )

        stream.unpause()

        while intrinsic_calibrator.grid_count < grid_count:
            logger.info(f"Waiting for sufficient calibration boards to become populated at port {port}")
            sleep(2)
        
        intrinsic_calibrator.calibrate_camera()