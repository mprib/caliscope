#%%
# This is a scratchpad for me to work through the data processing to transform points.csv
# into a PointHistory data object and then perform stereotriangulation on it given a CameraArray
from itertools import combinations
import pandas as pd
import numpy as np
from pathlib import Path
import time

from calicam import __root__
from calicam.cameras.data_packets import PointPacket, FramePacket, SyncPacket
from calicam.cameras.camera_array_builder import CameraArrayBuilder
from calicam.cameras.camera_array import CameraArray
from calicam.triangulate.triangulator import ArrayTriangulator

from calicam.triangulate.paired_point_builder import (
    StereoPointBuilder,
    StereoPointsPacket,
    SynchedStereoPointsPacket,
)

session_path = Path(__root__, "tests", "5_cameras")
point_data_path = Path(__root__, "tests", "5_cameras", "recording", "point_data.csv")
point_data = pd.read_csv(point_data_path)

xy_sync_indices = point_data["sync_index"].to_numpy()
xy_camera_indices = point_data["port"].to_numpy()
xy_point_id = point_data["point_id"].to_numpy()

img_x = point_data["img_loc_x"].to_numpy()
img_y = point_data["img_loc_y"].to_numpy()
xy_img = np.vstack([img_x, img_y]).T


ports = np.unique(xy_camera_indices)

paired_point_builder = StereoPointBuilder(ports)

sync_indices = np.unique(xy_sync_indices)
pairs = [(i, j) for i, j in combinations(ports, 2) if i < j]

# Create the infrastructure for the pairwise triangulation
camera_array: CameraArray = CameraArrayBuilder(
    Path(session_path, "config.toml")
).get_camera_array()
array_triangulator = ArrayTriangulator(camera_array)

stereotriangulated_table = None

print(time.time())
# for sync_index in [0,1,2,3]:
for sync_index in sync_indices:
    # pull in the data that shares the same sync index
    port_points = point_data.query(f"sync_index == {sync_index}")

    # initialize a dict to hold all the frame packets
    frame_packets = {}

    for port in ports:
        # Create the Frame packet for each port at this sync_index
        # and roll up into dictionary
        if port in port_points["port"].unique():
            points = port_points.query(f"port == {port}")
            frame_time = points["frame_time"].iloc[0]
            frame_index = points["frame_index"].iloc[0]

            point_id = points["point_id"].to_numpy()

            img_loc_x = points["img_loc_x"].to_numpy()
            img_loc_y = points["img_loc_y"].to_numpy()
            img_loc = np.vstack([img_loc_x, img_loc_y]).T

            board_loc_x = points["board_loc_x"].to_numpy()
            board_loc_y = points["board_loc_y"].to_numpy()
            board_loc = np.vstack([board_loc_x, board_loc_y]).T

            point_packet = PointPacket(point_id, img_loc, board_loc)
            frame_packet = FramePacket(
                port, frame_time, None, frame_index, point_packet
            )
            frame_packets[port] = frame_packet
        else:
            frame_packets[port] = None

    # create the sync packet for this sync index
    sync_packet = SyncPacket(sync_index, frame_packets)

    # get the paired point packets for all port pairs at this sync index
    synched_paired_points: SynchedStereoPointsPacket = (
        paired_point_builder.get_synched_paired_points(sync_packet)
    )
    # print(synched_paired_points)
    array_triangulator.triangulate_synched_points(synched_paired_points)

    for pair in synched_paired_points.pairs:
        triangulated_pair: StereoPointsPacket = (
            synched_paired_points.stereo_points_packets[pair]
        )
        if triangulated_pair is not None:
            if stereotriangulated_table is None:
                stereotriangulated_table = triangulated_pair.to_table()
            else:
                new_table = triangulated_pair.to_table()
                for key, value in new_table.items():
                    stereotriangulated_table[key].extend(value)
print(time.time())

#%%

stereotriangulated_table = pd.DataFrame(stereotriangulated_table)
stereotriangulated_table.to_csv(
    Path(session_path, "recording", "stereotriangulated_points.csv")
)
# %%
