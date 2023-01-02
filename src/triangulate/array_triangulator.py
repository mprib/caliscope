"""
This class will serve as the primary coordinator of the various camera pairs.
This will only deal with the CameraData object and not the Camera itself, which 
includes the additional functionality of overhead related to managing
the videocapture device.
"""
import logging

LOG_FILE = r"log\array_triangulator.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from itertools import combinations
from threading import Thread, Event
from queue import Queue

from src.cameras.camera_array import CameraArray, CameraArrayBuilder
from src.cameras.synchronizer import Synchronizer
from src.recording.recorded_stream import RecordedStreamPool
from src.triangulate.stereo_triangulator import StereoTriangulator
from src.triangulate.paired_point_stream import PairedPointStream

class ArrayTriangulator:
    
    def __init__(self, camera_array:CameraArray, paired_point_stream: PairedPointStream ):
        self.camera_array = camera_array
        self.paired_point_stream = paired_point_stream
        
        self.ports = [key for key in self.camera_array.cameras.keys()]
        
        # initialize stereo triangulators for each pair        
        self.stereo_triangulators = {}
        self.paired_point_qs = {}   # means to pass stereotriangulators new paired points to process
        for pair in self.paired_point_stream.pairs:
            logging.info(f"Creating StereoTriangulator for camera pair {pair}")
            portA = pair[0]
            portB = pair[1]
            camA = self.camera_array.cameras[portA]
            camB = self.camera_array.cameras[portB]
            pair_q = Queue(-1)
            self.paired_point_qs[pair] = pair_q
            self.stereo_triangulators[pair] = StereoTriangulator(camA, camB, pair_q)

        self.stop = Event() 

        self.thread = Thread(target=self.triangulate_points_worker, args=[], daemon=False)
        self.thread.start()
        
    def triangulate_points_worker(self):
        
        while not self.stop.is_set():
            new_paired_point_packet = self.paired_point_stream.out_q.get()
            self.paired_point_qs[new_paired_point_packet.pair].put(new_paired_point_packet)
            
            for pair, triangulator in self.stereo_triangulators.items():
                if not triangulator.out_q.empty():
                    triangulated_packet = triangulator.out_q.get()
                    print(pair)
                    print(triangulated_packet.xyz)

if __name__ == "__main__":
    from pathlib import Path 
    from src.calibration.charuco import Charuco
    from src.calibration.corner_tracker import CornerTracker

    # Build camera array from stored config file
    repo = str(Path(__file__)).split("src")[0]
    session_directory = Path(repo, "sessions", "iterative_adjustment")
    array_builder = CameraArrayBuilder(session_directory)
    camera_array = array_builder.get_camera_array()  

    
    # Build streams from pre-recorded video
    ports = [0,1,2]
    recorded_stream_pool = RecordedStreamPool(ports, session_directory)
    
    recorded_stream_pool.play_videos()
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None) # no fps target b/c not playing back for visual display

    # create a corner tracker to locate board corners
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    trackr = CornerTracker(charuco)

    # create a commmon point finder to grab charuco corners shared between the pair of ports
    pairs = [(0, 1), (0,2), (1,2)]
    point_stream = PairedPointStream(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,
    )

    # Build triangulator    
    array_triangulator = ArrayTriangulator(camera_array, point_stream)