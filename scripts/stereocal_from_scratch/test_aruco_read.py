import cv2


import logging
import time

from caliscope import __root__
from caliscope.logger import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# Copy over test data into a "working" script fixture directory to muck around
test_data_dir = __root__ / "tests/sessions/post_optimization"
recording_dir = test_data_dir / "calibration/extrinsic"
fixture_dir = __root__ / "scripts/fixtures/extrinsic_cal_sample"
# copy_contents(test_data_dir, fixture_dir)

calibration_video_dir = fixture_dir / "calibration/extrinsic"
sample_video_path = calibration_video_dir / "port_0.mp4"

capture = cv2.VideoCapture(str(sample_video_path))


while capture.isOpened():
    success, frame = capture.read()

    if success:
        cv2.imshow("press q to quit", frame)
        time.sleep(1 / 20)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
