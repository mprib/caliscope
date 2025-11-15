"""
The intent of this test script is to simply read aruco markers
allowing familiarization with the return signature of the detector.
A next step for me will be creating a Tracker for arucos that could
slot in to the overall pipeline.

Once I have that, then I can turn my attention to the stereocalibration
via alternate methods...And that would be just based on the raw text data,
so not really getting in the weeds regarding video processing.


"""

import cv2

import logging
import time

from caliscope import __root__
from caliscope.logger import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def process_aruco():
    # Copy over test data into a "working" script fixture directory to muck around
    test_data_dir = __root__ / "tests/sessions/post_optimization"
    test_data_dir / "calibration/extrinsic"
    fixture_dir = __root__ / "scripts/fixtures/extrinsic_cal_sample"
    # copy_contents(test_data_dir, fixture_dir)

    calibration_video_dir = fixture_dir / "calibration/extrinsic"
    sample_video_path = calibration_video_dir / "port_0.mp4"

    capture = cv2.VideoCapture(str(sample_video_path))

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary)

    while capture.isOpened():
        success, frame = capture.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
            frame = cv2.bitwise_not(frame)  # invert

            time.sleep(1 / 20)
            corners, ids, rejected = aruco_detector.detectMarkers(frame)

            # log
            if ids is not None:
                logger.info(f"detected {len(ids)} markers: {ids.flatten()}")
            else:
                logger.info(f"Found {len(rejected)} rejected candidates")

            # draw results for visualization
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 0, 255))
            else:
                # draw rejected candidates in red
                cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))

            cv2.imshow("press q to quit", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_aruco()
