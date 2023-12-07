import pyxy3d.logger

import cv2
from pathlib import Path
from pyxy3d.interface import FramePacket
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.cameras.camera_array import CameraData
from pyxy3d.interface import Tracker
from pyxy3d.gui.frame_emitters.playback_frame_emitter import PlaybackFrameEmitter
from pyxy3d.calibration.intrinsic_calibrator import IntrinsicCalibrator

logger = pyxy3d.logger.get(__name__)


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
            self.frame_emitters[camera.port] = PlaybackFrameEmitter(stream)
            self.frame_emitters[camera.port].start()
            self.calibrators[camera.port] = IntrinsicCalibrator(camera,stream)
   
   
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
             
    def get_frame_count(self,port):
        """
        Note that if frame_index may not start at 0 if frame_history is from a real-time capture
        """
        start_frame_index = self.streams[port].start_frame_index
        last_frame_index = self.streams[port].last_frame_index

        return last_frame_index - start_frame_index + 1
    
    def update_charuco(self,charuco_tracker:CharucoTracker):
        for stream in self.streams.values():
            stream.tracker = charuco_tracker
        
        for emitter in self.frame_emitters.values():
            emitter.initialize_grid_capture_history()
     
    def play_stream(self,port):
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
        self.calibrators[port].add_calibration_frame_indices(frame_index)
        new_ids = self.calibrators[port].all_ids[frame_index]
        new_img_loc = self.calibrators[port].all_img_loc[frame_index]
        self.frame_emitters[port].add_to_grid_history(new_ids, new_img_loc)

    def clear_calibration_data(self, port: int):
        self.calibrators[port].clear_calibration_data()
        self.frame_emitters[port].initialize_grid_capture_history()

    def calibrate_camera(self,port:int):
        logger.info(f"Calibrating camera at port {port}")
        self.calibrators[port].calibrate_camera()

    def apply_distortion(self, camera:CameraData, undistort: bool):
        self.frame_emitters[camera.port].update_distortion_params(
            undistort=undistort, 
            matrix = camera.matrix, 
            distortions = camera.distortions
        )

    def set_stream_rotation(self, port, rotation_count):
        self.streams[port].rotation_count = rotation_count