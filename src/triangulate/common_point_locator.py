from queue import Queue
import cv2

class PairedPointsLocator:
    
    
    def __init__(self, synchronizer):
        self.synchronizer = synchronizer
        



if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.recording.recorded_stream import RecordedStreamPool

    import time
    from src.cameras.synchronizer import Synchronizer
    
    repo = Path(__file__).parent.parent.parent
    print(repo)
    video_directory = Path(repo, "src", "triangulate", "sample_data", "stereo_track_charuco")

    ports = [0,1]
    recorded_stream_pool = RecordedStreamPool(ports, video_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos() 
    
    notification_q = Queue()
    syncr.subscribers.append(notification_q)

    while True:
        frame_bundle_notice = notification_q.get()
        for port, frame_data in syncr.current_bundle.items():
            if frame_data:
                cv2.imshow(f"Port {port}", frame_data["frame"])

        key = cv2.waitKey(1)

        if key == ord("q"):
            cv2.destroyAllWindows()
            break