import logging
import time
from caliscope import __root__
from caliscope.logger import setup_logging
from caliscope.configurator import Configurator
from caliscope.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.calibration.charuco import Charuco

setup_logging()

logger = logging.getLogger(__name__)

version = "post_optimization"
test_data_dir = __root__ / "tests/sessions/post_optimization"
recording_dir = test_data_dir / "calibration/extrinsic"

fixture_dir = __root__ / "scripts/fixtures/extrinsic_cal_sample"
calibration_video_dir = fixture_dir / "calibration/extrinsic"
# copy_contents(test_data_dir, fixture_dir)

config = Configurator(test_data_dir)
camera_array = config.get_camera_array()


charuco = Charuco(
    columns=4,
    rows=5,
    board_height=11.0,
    board_width=8.5,
    dictionary="DICT_4X4_1000",
    units="inch",
    aruco_scale=0.75,
    square_size_overide_cm=5.4,
    inverted=True,
)

charuco_tracker = CharucoTracker(charuco)


sync_stream_manager = SynchronizedStreamManager(
    recording_dir=calibration_video_dir, all_camera_data=camera_array.cameras, tracker=charuco_tracker
)

sync_stream_manager.process_streams(fps_target=100, include_video=True)

target_output_file = calibration_video_dir / f"{charuco_tracker.name}/xy_{charuco_tracker.name}.csv"

if target_output_file.exists():
    target_output_file.unlink()

# KIMI: file path messed up here.. I can see it exists...
while not target_output_file.exists():
    time.sleep(0.5)
    logger.info(f"Waiting for {target_output_file}")


print(test_data_dir)
print("end")
