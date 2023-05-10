import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from time import sleep
from queue import Queue
import cv2

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
from pyxy3d.trackers.tracker_enum import Tracker
# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents


def create_xy(
    config: Configurator,
    recording_path: Path,
    tracker_enum: Tracker,
):
    frame_times = pd.read_csv(Path(recording_path, "frame_time_history.csv"))
    sync_index_count = len(frame_times["sync_index"].unique())

    output_suffix = tracker_enum.name
    
    stream_pool = RecordedStreamPool(
        directory=recording_path,
        config=config,
        fps_target=100,
        tracker=tracker_enum,
    )

    synchronizer = Synchronizer(stream_pool.streams, fps_target=100)
    video_recorder = VideoRecorder(synchronizer, suffix=output_suffix)
    video_recorder.start_recording(
        destination_folder=recording_path,
        include_video=True,
        show_points=True,
    )
    stream_pool.play_videos()

    while video_recorder.recording:
        sleep(1)
        percent_complete = int((video_recorder.sync_index / sync_index_count) * 100)
        logger.info(f"{percent_complete}% processed")


def triangulate_xy_data(
    xy_data: pd.DataFrame, camera_array: CameraArray
) -> Dict[str, List]:
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

    for index in xy_data["sync_index"].unique():
        active_index = xy_data["sync_index"] == index
        cameras = xy_data["port"][active_index].to_numpy()
        point_ids = xy_data["point_id"][active_index].to_numpy()
        img_loc_x = xy_data["img_loc_x"][active_index].to_numpy()
        img_loc_y = xy_data["img_loc_y"][active_index].to_numpy()
        imgs_xy = np.vstack([img_loc_x, img_loc_y]).T

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

    return xyz_history



def create_xyz(session_path:Path, recording_path:Path, tracker_enum:Tracker)->None:
    """
    creates xyz_{tracker name}.csv file within the recording_path directory
    """
    config = Configurator(session_path)
    
    output_suffix = tracker_enum.name
    
    # locate xy_{tracker name}.csv
    xy_csv_path = Path(recording_path, f"xy_{output_suffix}.csv")
   
    # create if it doesn't already exist 
    if not xy_csv_path.exists():
       create_xy(config, recording_path, tracker_enum)
    
    # load in 2d data and triangulate it 
    xy_data = pd.read_csv(xy_csv_path)
    xyz_history = triangulate_xy_data(xy_data, config.get_camera_array())
    xyz_data = pd.DataFrame(xyz_history)
    xyz_data.to_csv(Path(recording_path, f"xyz_{output_suffix}.csv"))
    
    