# the PointHistory object requires a highly particular format
# all of the data is ultimately embedded in the initial camera array configuration
# and the calibration point data. These functions transform those two
# things into a PointHistory object that can be used to optimize the CaptureVolume
import caliscope.logger


import pandas as pd
import numpy as np
from pathlib import Path

from caliscope.packets import PointPacket, FramePacket, SyncPacket
from caliscope.cameras.camera_array import CameraArray
from caliscope.triangulate.array_stereo_triangulator import ArrayStereoTriangulator

from caliscope.triangulate.stereo_points_builder import (
    StereoPointsBuilder,
    StereoPointsPacket,
    SynchedStereoPointsPacket,
)
logger = caliscope.logger.get(__name__)


def get_stereotriangulated_table(
    camera_array: CameraArray, point_data_path: Path
) -> pd.DataFrame:
    logger.info(
        f"Beginning to create stereotriangulated points from data stored at {point_data_path}"
    )
    point_data = pd.read_csv(point_data_path)

    xy_sync_indices = point_data["sync_index"].to_numpy()
    sync_indices = np.unique(xy_sync_indices)

    xy_camera_indices = point_data["port"].to_numpy()
    # ports = np.unique(xy_camera_indices)
    ports = [key for key in camera_array.port_index.keys()]

    paired_point_builder = StereoPointsBuilder(ports)

    # Create the infrastructure for the pairwise triangulation
    array_triangulator = ArrayStereoTriangulator(camera_array)
    stereotriangulated_table = None

    logger.info("Begin reconstructing SyncPackets and SynchedStereoPairs... ")
    for sync_index in sync_indices:
        if sync_index % 25 == 0:
            logger.info(
                f"Processing stereotriangulation estimates...currently at sync index {sync_index}"
            )
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

                obj_loc_x = points["obj_loc_x"].to_numpy()
                obj_loc_y = points["obj_loc_y"].to_numpy()
                obj_loc = np.vstack([obj_loc_x, obj_loc_y]).T

                point_packet = PointPacket(point_id, img_loc, obj_loc)
                frame_packet = FramePacket(
                    port=port,
                    frame_index=frame_index,
                    frame_time=frame_time,
                    frame=None,
                    points=point_packet,
                )
                frame_packets[port] = frame_packet
            else:
                frame_packets[port] = None

        # create the sync packet for this sync index
        sync_packet = SyncPacket(sync_index, frame_packets)

        # get the paired point packets for all port pairs at this sync index
        synched_stereo_points: SynchedStereoPointsPacket = (
            paired_point_builder.get_synched_paired_points(sync_packet)
        )
        # print(synched_paired_points)
        array_triangulator.triangulate_synched_points(synched_stereo_points)

        for pair in synched_stereo_points.pairs:
            triangulated_pair: StereoPointsPacket = (
                synched_stereo_points.stereo_points_packets[pair]
            )
            if triangulated_pair is not None:
                if stereotriangulated_table is None:
                    stereotriangulated_table = triangulated_pair.to_table()
                else:
                    new_table = triangulated_pair.to_table()
                    for key, value in new_table.items():
                        stereotriangulated_table[key].extend(value)

    logger.info(
        f"Saving stereotriangulated_points.csv to {point_data_path.parent} for inspection"
    )
    stereotriangulated_table = pd.DataFrame(stereotriangulated_table)
    stereotriangulated_table.to_csv(
        Path(point_data_path.parent, "stereotriangulated_points.csv")
    )

    logger.info("Returning dataframe of stereotriangulated points to caller")

    return stereotriangulated_table
