"""
This class will serve as the primary coordinator of the various camera pairs.
This will only deal with the CameraData object and not the Camera itself, which 
includes the additional functionality of overhead related to managing
the videocapture device.

At the moment the priority is getting triangulated data saved out in a csv that could
then be played back into a visualizer, or used as the basis for an array config optimization.

"""
import calicam.logger

logger = calicam.logger.get(__name__)

from itertools import combinations
from threading import Thread, Event
from queue import Queue
import pandas as pd
from pathlib import Path

from calicam.cameras.camera_array import CameraArray
from calicam.cameras.camera_array_builder import CameraArrayBuilder
from calicam.cameras.synchronizer import Synchronizer
from calicam.recording.recorded_stream import RecordedStreamPool
from calicam.triangulate.stereo_triangulator import StereoTriangulator
from calicam.triangulate.paired_point_builder import (
    StereoPointBuilder,
    StereoPointsPacket,
)


class ArrayTriangulator:
    def __init__(
        self,
        camera_array: CameraArray,
        paired_point_stream: StereoPointBuilder,
        output_file: Path = None,
    ):
        self.camera_array = camera_array
        self.paired_point_stream = paired_point_stream
        self.output_file = output_file

        self.ports = [key for key in self.camera_array.cameras.keys()]

        # initialize stereo triangulators for each pair
        self.stereo_triangulators = {}
        self.paired_point_qs = (
            {}
        )  # means to pass stereotriangulators new paired points to process

        self.agg_3d_points = None

        for pair in self.paired_point_stream.pairs:
            logger.info(f"Creating StereoTriangulator for camera pair {pair}")
            portA = pair[0]
            portB = pair[1]
            camA = self.camera_array.cameras[portA]
            camB = self.camera_array.cameras[portB]
            self.stereo_triangulators[pair] = StereoTriangulator(
                camA,
                camB,
            )

        self.stop = Event()

        self.thread = Thread(
            target=self.triangulate_points_worker, args=[], daemon=False
        )
        self.thread.start()

    def store_point_data(self, packet):

        packet_dict = packet.to_dict()

        if self.agg_3d_points is None:
            # build a dictionary of lists that will form basis of dataframe output to csv
            self.agg_3d_points = {}
            for key, value in packet_dict.items():
                self.agg_3d_points[key] = []

        for key, value in packet_dict.items():
            self.agg_3d_points[key].extend(value)

    def triangulate_points_worker(self):

        while not self.stop.is_set():
            # read in a paired point stream
            new_paired_point_packet: StereoPointsPacket = (
                self.paired_point_stream.out_q.get()
            )

            if new_paired_point_packet is None:
                logger.info(
                    "`None` detected on paired points stream...shutting down triangulator"
                )
                self.stop.set()

            else:
                logger.info(
                    f"Sync Index: {new_paired_point_packet.sync_index} | Pair: {new_paired_point_packet.pair}"
                )

                pair = new_paired_point_packet.pair
                triangulated_packet = self.stereo_triangulators[pair].get_3D_points(
                    new_paired_point_packet
                )

                self.store_point_data(triangulated_packet)

                print(f"Sync Index: {triangulated_packet.sync_index}  Pair: {pair}")

        # after processing is complete, save out the data   
        if self.output_file is not None:
            logger.info(f"Saving triangulated point csv to {self.output_file}")
            self.agg_3d_points = pd.DataFrame(self.agg_3d_points)
            self.agg_3d_points.to_csv(self.output_file)


if __name__ == "__main__":
    from pathlib import Path
    from calicam.calibration.charuco import Charuco
    from calicam.calibration.corner_tracker import CornerTracker
    from calicam import __root__

    session_directory = Path(__root__, "tests", "5_cameras")
    config_path = Path(session_directory, "config.toml")
    array_builder = CameraArrayBuilder(config_path)
    camera_array = array_builder.get_camera_array()

    # Build streams from pre-recorded video
    recording_directory = Path(session_directory, "recording")
    ports = [0, 1, 2, 3, 4]
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    recorded_stream_pool = RecordedStreamPool(
        ports, recording_directory, charuco=charuco
    )

    # synchronize videos
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=100)
    recorded_stream_pool.play_videos()

    # create a commmon point finder to grab charuco corners shared between the pair of ports
    point_stream = StereoPointBuilder(synchronizer=syncr)

    # Build triangulator
    # Note that this will automatically create the summarized output of the projected points
    # this is just a temporary setup while I try to figure out something more suitable long-term

    output_file = Path(recording_directory, "triangulated_points.csv")
    array_triangulator = ArrayTriangulator(camera_array, point_stream, output_file)
