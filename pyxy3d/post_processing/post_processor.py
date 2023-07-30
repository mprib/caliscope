import typing
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from time import sleep, time
from queue import Queue
import cv2
from PySide6.QtCore import QObject, Signal

import sys
from PySide6.QtWidgets import QApplication
from pyxy3d.configurator import Configurator
from pathlib import Path
import numpy as np
from numba.typed import Dict, List
from pyxy3d import __root__
import pandas as pd
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.recording.recorded_stream import RecordedStream, RecordedStreamPool
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.recording.video_recorder import VideoRecorder
from pyxy3d.triangulate.triangulation import triangulate_xy

from pyxy3d.interface import FramePacket, Tracker
from pyxy3d.trackers.tracker_enum import TrackerEnum

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents
from pyxy3d.export import xyz_to_trc
from pyxy3d.post_processing.gap_filling import gap_fill_xy, gap_fill_xyz
from pyxy3d.post_processing.smoothing import smooth_xyz

class PostProcessor:
    """
    The post processer operates independently of the session. It does not need to worry about camera management.
    Provide it with a path to the directory that contains the following:
    - config.toml
    - frame_time.csv 
    - .mp4 files
    

    """
    # progress_update = Signal(dict)  # {"stage": str, "percent":int}

    def __init__(self,recording_path:Path, tracker_enum:TrackerEnum):
        self.recording_path = recording_path
        self.tracker_enum = tracker_enum
        self.config = Configurator(self.recording_path)
        self.camera_array = self.config.get_camera_array()
        self.fps = self.config.get_fps_recording()

    def create_xy(self):
        """
        Reads through all .mp4  files in the recording path and applies the tracker to them
        The xy_TrackerName.csv file is saved out to the same directory by the VideoRecorder
        """
        frame_times = pd.read_csv(Path(self.recording_path, "frame_time_history.csv"))
        sync_index_count = len(frame_times["sync_index"].unique())

        fps_recording = self.config.get_fps_recording()
        logger.info("Creating pool of playback streams to begin processing")
        stream_pool = RecordedStreamPool(
            directory=self.recording_path,
            config=self.config,
            fps_target=fps_recording,
            tracker=self.tracker_enum.value(),
        )

        synchronizer = Synchronizer(stream_pool.streams, fps_target=fps_recording)

        logger.info(
            "Creating video recorder to record (x,y) data estimates from PointPacket delivered by Tracker"
        )
        output_suffix = self.tracker_enum.name
        
        # it is the videorecorder that will save the (x,y) landmark positionsj
        video_recorder = VideoRecorder(synchronizer, suffix=output_suffix)

        # these (x,y) positions will be stored within the subdirectory of the recording folder
        # this destination subfolder is named to align with the tracker_enum.name
        destination_folder = Path(self.recording_path, self.tracker_enum.name)
        video_recorder.start_recording(
            destination_folder=destination_folder,
            include_video=True,
            show_points=True,
            store_point_history=True
        )
        logger.info("Initiate playback and processing")
        stream_pool.play_videos()

        while video_recorder.recording:
            sleep(1)
            percent_complete = int((video_recorder.sync_index / sync_index_count) * 100)
            logger.info(f"(Stage 1 of 2): {percent_complete}% of frames processed for (x,y) landmark detection")

    def create_xyz(self, xy_gap_fill = 3, xyz_gap_fill = 3, cutoff_freq = 6, include_trc = True) -> None:
        """
        creates xyz_{tracker name}.csv file within the recording_path directory

        Uses the two functions above, first creating the xy points based on the tracker if they 
        don't already exist, the triangulating them. Makes use of an internal method self.triangulate_xy_data
        
        """

        output_suffix = self.tracker_enum.name

        tracker_output_path = Path(self.recording_path, self.tracker_enum.name)
        # locate xy_{tracker name}.csv
        xy_csv_path = Path(tracker_output_path, f"xy_{output_suffix}.csv")

        # create if it doesn't already exist
        if not xy_csv_path.exists():
            self.create_xy()

        # load in 2d data and triangulate it
        logger.info("Reading in (x,y) data..")
        xy = pd.read_csv(xy_csv_path)
        logger.info("Filling small gaps in (x,y) data")
        xy = gap_fill_xy(xy)
        logger.info("Beginning data triangulation")
        xyz = triangulate_xy(self.camera_array, xy)
        logger.info("Filling small gaps in (x,y,z) data")
        xyz = gap_fill_xyz(xyz)
        logger.info(f"Smoothing (x,y,z) using butterworth filter with cutoff frequency of 6hz")
        xyz = smooth_xyz(xyz, order=2, fps=self.fps, cutoff=cutoff_freq)
        logger.info("Saving (x,y,z) to csv file")       
        xyz_csv_path = Path(tracker_output_path, f"xyz_{output_suffix}.csv")
        xyz.to_csv(xyz_csv_path)

        # only include trc if wanted and only if there is actually good data to export
        if include_trc and xyz.shape[0] > 0:
           xyz_to_trc(xyz_csv_path, tracker = self.tracker_enum.value()) 

