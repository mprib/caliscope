"""
This class will serve as the primary coordinator of the various camera pairs.
This will only deal with the CameraData object and not the Camera itself, which 
includes the additional functionality of overhead related to managing
the videocapture device.

At the moment the priority is getting triangulated data saved out in a csv that could
then be played back into a visualizer, or used as the basis for an array config optimization.

"""
import logging

LOG_FILE = r"log\array_triangulator.log"
LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from itertools import combinations
from threading import Thread, Event
from queue import Queue
import pandas as pd

from src.cameras.camera_array import CameraArray, CameraArrayBuilder
from src.cameras.synchronizer import Synchronizer
from src.recording.recorded_stream import RecordedStreamPool
from src.triangulate.stereo_triangulator import StereoTriangulator
from src.triangulate.paired_point_stream import PairedPointStream


class ArrayTriangulator:
    def __init__(
        self,
        camera_array: CameraArray,
        paired_point_stream: PairedPointStream,
        save_directory=None,
    ):
        self.camera_array = camera_array
        self.paired_point_stream = paired_point_stream
        self.save_directory = save_directory

        self.ports = [key for key in self.camera_array.cameras.keys()]

        # initialize stereo triangulators for each pair
        self.stereo_triangulators = {}
        self.paired_point_qs = (
            {}
        )  # means to pass stereotriangulators new paired points to process
        for pair in self.paired_point_stream.pairs:
            logging.info(f"Creating StereoTriangulator for camera pair {pair}")
            portA = pair[0]
            portB = pair[1]
            camA = self.camera_array.cameras[portA]
            camB = self.camera_array.cameras[portB]
            # pair_q = Queue(-1)
            self.paired_point_qs[pair] = Queue(-1)
            self.stereo_triangulators[pair] = StereoTriangulator(
                camA, camB, self.paired_point_qs[pair]
            )

        self.stop = Event()

        self.thread = Thread(
            target=self.triangulate_points_worker, args=[], daemon=False
        )
        self.thread.start()

    def triangulate_points_worker(self):
        # build a dictionary of lists that will form basis of dataframe output to csv
        aggregate_3d_points = {
            "pair": [],
            "time": [],
            "bundle": [],
            "id": [],
            "x_pos": [],
            "y_pos": [],
            "z_pos": [],
        }

        while not self.stop.is_set():
            # read in a paired point stream
            new_paired_point_packet = self.paired_point_stream.out_q.get()
            logging.info(
                f"Bundle: {new_paired_point_packet.bundle_index} | Pair: {new_paired_point_packet.pair}"
            )
            # hand off the paired points to the appropriate triangulator via queue
            # honestly, the more I look at this the more I hate it. make it explicit
            # that you are handing off to a stereo triangulator
            pair = new_paired_point_packet.pair
            self.paired_point_qs[pair].put(new_paired_point_packet)
            triangulated_packet = self.stereo_triangulators[pair].out_q.get()

            # for pair, triangulator in self.stereo_triangulators.items():
            #     if not triangulator.out_q.empty():
            #         triangulated_packet = triangulator.out_q.get()
            packet_data = triangulated_packet.to_dict()

            for key, value in packet_data.items():
                aggregate_3d_points[key].extend(value)

            print(f"Bundle: {triangulated_packet.bundle_index}  Pair: {pair}")
            # TODO: #45 figure out how to get this to stop automatically
            # might want to get the frame counts for the saved port data
            if triangulated_packet.bundle_index > 30:
                self.stop.set()

        if self.save_directory is not None:
            logging.info(f"Saving triangulated point csv to {self.save_directory}")
            aggregate_3d_points = pd.DataFrame(aggregate_3d_points)
            aggregate_3d_points.to_csv(
                Path(self.save_directory, "triangulated_points.csv")
            )


if __name__ == "__main__":
    from pathlib import Path
    from src.calibration.charuco import Charuco
    from src.calibration.corner_tracker import CornerTracker

    # Build camera array from stored config file
    repo = str(Path(__file__)).split("src")[0]
    session_directory = Path(repo, "sessions", "iterative_adjustment")
    config_file = Path(session_directory, "config.toml")
    array_builder = CameraArrayBuilder(config_file)
    camera_array = array_builder.get_camera_array()

    # Build streams from pre-recorded video
    ports = [0, 1, 2]
    recorded_stream_pool = RecordedStreamPool(ports, session_directory)

    recorded_stream_pool.play_videos()
    syncr = Synchronizer(
        recorded_stream_pool.streams, fps_target=None
    )  # no fps target b/c not playing back for visual display

    # create a corner tracker to locate board corners
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    trackr = CornerTracker(charuco)

    # create a commmon point finder to grab charuco corners shared between the pair of ports
    pairs = [(0, 1), (0, 2), (1, 2)]
    point_stream = PairedPointStream(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,
    )

    # Build triangulator
    # Note that this will automatically create the summarized output of the projected points
    # this is just a temporary setup while I try to figure out something more suitable long-term
    array_triangulator = ArrayTriangulator(
        camera_array, point_stream, session_directory
    )
