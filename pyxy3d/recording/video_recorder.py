import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from pathlib import Path
from queue import Queue
from threading import Thread, Event
import cv2
import sys
import pandas as pd

from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.cameras.live_stream import LiveStream

class VideoRecorder:
    def __init__(self, synchronizer:Synchronizer):
        self.synchronizer = synchronizer

        self.recording = False
        # build dict that will be stored to csv
        self.trigger_stop = Event()

    def build_video_writers(self):

        # create a dictionary of videowriters
        self.video_writers = {}
        for port, stream in self.synchronizer.streams.items():

            path = str(Path(self.destination_folder, f"port_{port}.mp4"))
            logger.info(f"Building video writer for port {port}; recording to {path}")
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            fps = self.synchronizer.get_fps_target()
            frame_size = stream.camera.size

            writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
            self.video_writers[port] = writer

    def save_frame_worker(self):
        # connect video recorder to synchronizer via an "in" queue
        self.build_video_writers()

        # I think I put this here so that it will get reset if you reuse the same recorder..
        self.frame_history = {
            "sync_index": [],
            "port": [],
            "frame_index": [],
            "frame_time": [],
        }
        self.point_data_history = {
                    "sync_index":[],
                    "port":[],
                    "frame_index":[],
                    "frame_time":[],
                    "point_id":[],
                    "img_loc_x":[],
                    "img_loc_y":[],
                    "board_loc_x":[],
                    "board_loc_y":[],
        }
        
        self.sync_packet_in_q = Queue(-1)
        self.synchronizer.subscribe_to_sync_packets(self.sync_packet_in_q)
        
        # reset in case recording a second time
        # self.trigger_stop.clear() 
        
        while not self.trigger_stop.is_set():
            sync_packet = self.sync_packet_in_q.get()
            logger.debug("Pulling sync packet from queue")
            sync_index = sync_packet.sync_index
            
            for port, frame_packet in sync_packet.frame_packets.items():
                if frame_packet is not None:
                    # logger.info("Processiong frame packet...")
                    # read in the data for this frame for this port
                    frame = frame_packet.frame
                    frame_index = frame_packet.frame_index
                    frame_time = frame_packet.frame_time

                    # store the frame
                    self.video_writers[port].write(frame)

                    # store to assocated data in the dictionary
                    self.frame_history["sync_index"].append(sync_index)
                    self.frame_history["port"].append(port)
                    self.frame_history["frame_index"].append(frame_index)
                    self.frame_history["frame_time"].append(frame_time)
                    
                    new_tidy_table = frame_packet.to_tidy_table(sync_index)
                    if new_tidy_table is not None: # i.e. it has data
                        for key, value in self.point_data_history.copy().items():
                            self.point_data_history[key].extend(new_tidy_table[key])
                        print(new_tidy_table)

        logger.info("Save frame worker winding down...")
        self.synchronizer.release_sync_packet_q(self.sync_packet_in_q)

        # a proper release is strictly necessary to ensure file is readable
        logger.info("releasing video writers...")
        for port, frame_packet in sync_packet.frame_packets.items():
            self.video_writers[port].release()

        logger.info("Initiate storing of frame history")
        self.store_frame_history()
        logger.info("Initiate storing of point history")
        self.store_point_history()
        self.trigger_stop.clear()  # reset stop recording trigger
        self.recording = False
        del self.video_writers
                
    def store_point_history(self):
        df = pd.DataFrame(self.point_data_history)
        # TODO: #25 if file exists then change the name
        point_data_path = str(Path(self.destination_folder, "point_data.csv"))
        logger.info(f"Storing point data in {point_data_path}")
        df.to_csv(point_data_path, index=False, header=True)

    def store_frame_history(self):
        df = pd.DataFrame(self.frame_history)
        # TODO: #25 if file exists then change the name
        frame_hist_path = str(Path(self.destination_folder, "frame_time_history.csv"))
        logger.info(f"Storing frame history to {frame_hist_path}")
        df.to_csv(frame_hist_path, index=False, header=True)

    def start_recording(self, destination_folder:Path):

        logger.info(f"All video data to be saved to {destination_folder}")

        self.destination_folder = destination_folder
        # create the folder if it doesn't already exist
        self.destination_folder.mkdir(exist_ok=True, parents=True)

        self.recording = True
        self.recording_thread = Thread(
            target=self.save_frame_worker, args=[], daemon=True
        )
        self.recording_thread.start()

    def stop_recording(self):
        logger.info("Stop recording initiated within VideoRecorder")
        self.trigger_stop.set()


if __name__ == "__main__":

    import time

    from pyxy3d.cameras.camera import Camera
    from pyxy3d.cameras.live_stream import LiveStream
    from pyxy3d.session import Session
    from pyxy3d.calibration.charuco import Charuco

    from pyxy3d.recording.recorded_stream import RecordedStream, RecordedStreamPool
    from pyxy3d import __root__


    ports = [0, 1, 2, 3, 4]
    # ports = [0,1]

    test_live = True
    # test_live = False

    if test_live:

        session_directory = Path(__root__, "tests", "please work")
        session = Session(session_directory)
        session.load_cameras()
        session.load_streams()
        session.adjust_resolutions()

        for port, stream in session.streams.items():
            stream._show_fps = True
            # stream._show_charuco = True

        logger.info("Creating Synchronizer")
        syncr = Synchronizer(session.streams, fps_target=30)
        video_path = Path(session_directory, "recording2")
    else:
        recording_directory = Path(__root__, "sessions", "5_cameras", "recording")
        charuco = Charuco(
            4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
        )
        stream_pool = RecordedStreamPool(ports, recording_directory, charuco=charuco)
        logger.info("Creating Synchronizer")
        syncr = Synchronizer(stream_pool.streams, fps_target=3)
        stream_pool.play_videos()
        new_recording_directory = Path(__root__, "sessions", "5_cameras", "recording2")
        video_path = Path(new_recording_directory)

    video_recorder = VideoRecorder(syncr)

    video_recorder.start_recording(video_path)
    time.sleep(20)
    # while not syncr.stop_event.is_set():
    #     time.sleep(1)

    video_recorder.stop_recording()
