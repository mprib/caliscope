import pyxy3d.logger
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.trackers.charuco_tracker import CharucoTracker

logger = pyxy3d.logger.get(__name__)


class IntrinsicCalibrator:
    """
    Takes a recorded stream and determines a CameraData object from it 
    """ 
    def __init__(self, stream: RecordedStream, tracker:CharucoTracker) -> None:
        pass