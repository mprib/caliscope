from pathlib import Path
from caliscope import __root__
from caliscope.motion_trial import MotionTrial
from caliscope.packets import XYZPacket
import caliscope.logger

logger = caliscope.logger.get(__name__)

def test_motion_trial():
    
    test_csv = Path(__root__, "tests", "sessions", "4_cam_recording", "recording_1", "HOLISTIC_OPENSIM", "xyz_HOLISTIC_OPENSIM.csv")
    motion_trial = MotionTrial(test_csv)

    # able to read in xyz packet
    xyz_packet_0 = motion_trial.get_xyz(0)
    assert(isinstance(xyz_packet_0, XYZPacket))
    assert(xyz_packet_0.point_xyz.shape[1]==3)
    
    # able to read in tracker information
    assert(motion_trial.tracker.name == "HOLISTIC_OPENSIM")
    assert(xyz_packet_0.point_ids.shape[0] == xyz_packet_0.point_xyz.shape[0])

    logger.info(motion_trial)     

    # todo:
    # motion trial has active frame that defaults to 0 or last read frame

if __name__ == "__main__":
    test_motion_trial()
     


