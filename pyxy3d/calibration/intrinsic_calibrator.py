import pyxy3d.logger
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.trackers.charuco_tracker import CharucoTracker

logger = pyxy3d.logger.get(__name__)


class PreRecordedIntrinsicCalibrator:
    """
    Takes a recorded stream and determines a CameraData object from it 
    Stream needs to have a charuco tracker assigned to it
    """ 
    def __init__(self, stream: RecordedStream) -> None:
        pass