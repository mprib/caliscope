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
from pyxy3d.trackers.holistic_tracker import HolisticTrackerFactory
from pyxy3d.trackers.hand_tracker import HandTrackerFactory
from pyxy3d.trackers.pose_tracker import PoseTracker
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.recording.recorded_stream import RecordedStream, RecordedStreamPool
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.recording.video_recorder import VideoRecorder
from pyxy3d.triangulate.sync_packet_triangulator import SyncPacketTriangulator, triangulate_sync_index
from pyxy3d.interface import FramePacket, TrackerFactory

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents

# session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration_2_cam")
# copy_session_path = Path(
#     __root__, "tests", "sessions_copy_delete", "mediapipe_calibration_2_cam"
# )
# copy_contents(session_path, copy_session_path)


# config = Configurator(copy_session_path)


def create_xy_points(
    config: Configurator, recording_directory: Path, tracker_factory: TrackerFactory
):
    frame_times = pd.read_csv(Path(recording_directory, "frame_time_history.csv"))
    sync_index_count = len(frame_times["sync_index"].unique())

    stream_pool = RecordedStreamPool(
        directory=recording_directory,
        config=config,
        fps_target=100,
        tracker_factory=tracker_factory,
    )
    synchronizer = Synchronizer(stream_pool.streams, fps_target=100)
    video_recorder = VideoRecorder(synchronizer)
    video_recorder.start_recording(
        destination_folder=recording_directory,
        include_video=True,
        show_points=True,
        suffix="_xy",
    )
    stream_pool.play_videos()

    while video_recorder.recording:
        sleep(1)
        percent_complete = int((video_recorder.sync_index / sync_index_count) * 100)
        logger.info(f"{percent_complete}% processed")

def triangulate_xy_data(xy_data:pd.DataFrame, camera_array:CameraArray)->Dict[str,List]:

    # assemble numba compatible dictionary
    projection_matrices = Dict()
    for port, cam in camera_array.cameras.items():
        projection_matrices[int(port)] = cam.projection_matrix
    
    xyz_history = {"point_id":[],
                   "x_coord": [],
                   "y_coord": [],
                   "z_coord": [],}
    
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
            points_xyz = np.array(points_xyz)
            xyz_history["point_id"].extend(point_id_xyz)
            xyz_history["x_coord"].extend(points_xyz[:,0].tolist())
            xyz_history["y_coord"].extend(points_xyz[:,1].tolist())
            xyz_history["z_coord"].extend(points_xyz[:,2].tolist())

    return xyz_history