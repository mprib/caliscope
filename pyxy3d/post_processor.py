import typing
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from time import sleep, time
from queue import Queue
import cv2
from PyQt6.QtCore import QObject, pyqtSignal

import sys
from PyQt6.QtWidgets import QApplication
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
from pyxy3d.triangulate.sync_packet_triangulator import (
    SyncPacketTriangulator,
    triangulate_sync_index,
)
from pyxy3d.interface import FramePacket, Tracker
from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.gui.playback_widget import PlaybackWidget

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents


class PostProcessor(QObject):
    progress_update = pyqtSignal(dict)  # {"stage": str, "percent":int}

    def __init__(self,config:Configurator ):
        super().__init__()
        self.config = config

    def create_xyz(self, recording_path: Path, tracker_enum: TrackerEnum,) -> None:
        """
        creates xyz_{tracker name}.csv file within the recording_path directory

        Uses the two functions above, first creating the xy points based on the tracker if they 
        don't already exist, the triangulating them. Makes use of an internal method self.triangulate_xy_data
        
        """

        output_suffix = tracker_enum.name

        tracker_output_path = Path(recording_path, tracker_enum.name)
        # locate xy_{tracker name}.csv
        xy_csv_path = Path(tracker_output_path, f"xy_{output_suffix}.csv")

        # create if it doesn't already exist
        if not xy_csv_path.exists():
            self.create_xy(recording_path, tracker_enum)

        # load in 2d data and triangulate it
        xy_data = pd.read_csv(xy_csv_path)
        xyz_history = self.triangulate_xy_data(xy_data)
        xyz_data = pd.DataFrame(xyz_history)
        xyz_data.to_csv(Path(tracker_output_path, f"xyz_{output_suffix}.csv"))

    def create_xy(self, recording_path: Path, tracker_enum: TrackerEnum,):
        """
        Reads through all .mp4  files in the recording path and applies the tracker to them
        The xy_TrackerName.csv file is saved out to the same directory by the VideoRecorder
        """
        frame_times = pd.read_csv(Path(recording_path, "frame_time_history.csv"))
        sync_index_count = len(frame_times["sync_index"].unique())


        logger.info("Creating pool of playback streams to begin processing")
        stream_pool = RecordedStreamPool(
            directory=recording_path,
            config=self.config,
            fps_target=100,
            tracker=tracker_enum.value(),
        )

        synchronizer = Synchronizer(stream_pool.streams, fps_target=100)

        logger.info(
            "Creating video recorder to record (x,y) data estimates from PointPacket delivered by Tracker"
        )
        output_suffix = tracker_enum.name
        video_recorder = VideoRecorder(synchronizer, suffix=output_suffix)

        # store video files in a subfolder named by the tracker_enum.name
        destination_folder = Path(recording_path, tracker_enum.name)
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
            logger.info(f"{percent_complete}% processed")
            self.progress_update.emit(
                {
                    "stage": "Estimating (x,y) landmark positions (stage 1 of 2)",
                    "percent": percent_complete,
                }
            )


    def triangulate_xy_data(self, xy_data: pd.DataFrame) -> Dict[str, List]:
        
        camera_array = self.config.get_camera_array()
        # assemble numba compatible dictionary
        projection_matrices = Dict()
        for port, cam in camera_array.cameras.items():
            projection_matrices[int(port)] = cam.projection_matrix

        xyz_history = {
            "sync_index": [],
            "point_id": [],
            "x_coord": [],
            "y_coord": [],
            "z_coord": [],
        }

        sync_index_max = xy_data["sync_index"].max()

        start = time()
        last_log_update = int(start)  # only report progress each second

        for index in xy_data["sync_index"].unique():
            active_index = xy_data["sync_index"] == index
            cameras = xy_data["port"][active_index].to_numpy()
            point_ids = xy_data["point_id"][active_index].to_numpy()
            img_loc_x = xy_data["img_loc_x"][active_index].to_numpy()
            img_loc_y = xy_data["img_loc_y"][active_index].to_numpy()
            imgs_xy = np.vstack([img_loc_x, img_loc_y]).T

            # the fancy part
            point_id_xyz, points_xyz = triangulate_sync_index(
                projection_matrices, cameras, point_ids, imgs_xy
            )

            if len(point_id_xyz) > 0:
                # there are points to store so store them...
                xyz_history["sync_index"].extend([index] * len(point_id_xyz))
                xyz_history["point_id"].extend(point_id_xyz)

                points_xyz = np.array(points_xyz)
                xyz_history["x_coord"].extend(points_xyz[:, 0].tolist())
                xyz_history["y_coord"].extend(points_xyz[:, 1].tolist())
                xyz_history["z_coord"].extend(points_xyz[:, 2].tolist())

            # only log percent complete each second
            if int(time()) - last_log_update >= 1:
                percent_complete = int(100*(index/sync_index_max))
                logger.info(
                    f"Triangulation of (x,y) point estimates is {percent_complete}% complete"
                )
                last_log_update = int(time())
                self.progress_update.emit(
                    {
                        "stage": "Triangulating (x,y,z) estimates (stage 2 of 2)",
                        "percent": percent_complete,
                    }
                )

        # signalling the progress bar can now close
        self.progress_update.emit(
            {
                "stage": "Triangulating (x,y,z) estimates (stage 2 of 2)",
                "percent": 100,
                "close": True
            }
        )

        return xyz_history

