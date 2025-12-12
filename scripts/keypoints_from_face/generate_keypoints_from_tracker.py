"""
Integration script demonstrating ArUcoTracker with SynchronizedStreamManager.

This is a working reference implementation that will be purged once
ArUcoTracker is fully integrated into the main workflow.
"""

import logging
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication
import sys

from caliscope.cameras.camera_array import CameraArray
from caliscope.controller import Controller
from caliscope.configurator import Configurator
from caliscope.trackers.skull_tracker.skull_tracker import SkullTracker
from caliscope.logger import setup_logging
from caliscope.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.calibration.array_initialization.build_paired_pose_network import build_paired_pose_network
from caliscope.tracker import Tracker
from caliscope.post_processing.point_data import ImagePoints
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.gui.vizualize.calibration.capture_volume_widget import CaptureVolumeWidget
from caliscope.helper import copy_contents_to_clean_dest

setup_logging()
logger = logging.getLogger(__name__)


def generate_keypoints(camera_array: CameraArray, raw_video_dir: Path, tracker: Tracker):
    """Process calibration videos with Tracker to create xy_TRACKER.csv."""

    logger.info(f"Processing calibration videos with {tracker.name} tracker...")

    # Create synchronized stream manager
    sync_stream_manager = SynchronizedStreamManager(
        recording_dir=raw_video_dir, all_camera_data=camera_array.cameras, tracker=tracker
    )

    # Process streams (ArUco detection is fast, so high fps_target is fine)
    sync_stream_manager.process_streams(fps_target=100, include_video=True)
    # Wait for output file to be created
    target_output_file = raw_video_dir / f"{tracker.name}/xy_{tracker.name}.csv"

    time.time()
    while not target_output_file.exists():
        time.sleep(2)
        logger.info(f"Waiting for {target_output_file}")

    assert target_output_file.exists(), "Tracker output creation failed"

    logger.info(f"Successfully created: {target_output_file}")


if __name__ == "__main__":
    RERUN_KEYPOINT_GENERATION = False  # toggle during dev

    # This is Meero's config.toml and video files in the caliscope project structure
    # project root
    #   ├── config.toml
    #   ├── calibration
    #   │   ├── extrinsic
    #   │   │   └── port_1.mp4, port_2.mp4, port_3.mp4
    #   │   └── intrinsic
    #   └── recordings
    #       └── treadmill
    #           └── port_1.mp4, port_2.mp4, port_3.mp4
    origin_project_dir = Path("/home/mprib/caliscope_projects/markerless_calibration_data/caliscope_version")

    # destination to copy over origin clean project just for ease of reference and avoidance of pollution across runs
    working_project_dir = Path(__file__).parent / "sample_project"

    raw_video_dir = working_project_dir / "calibration/extrinsic"

    # this will create x,y image points along with x,y,z object points for pnp
    tracker = SkullTracker()

    keypoint_generation_time = None

    if RERUN_KEYPOINT_GENERATION:
        # create key points from clean project
        copy_contents_to_clean_dest(origin_project_dir, working_project_dir)

        # Load camera array from working project config
        config = Configurator(working_project_dir)
        camera_array = config.get_camera_array()

        # generate keypoints from tracker that yields object x,y,z points
        tic = time.time()
        generate_keypoints(camera_array, raw_video_dir, tracker)
        toc = time.time()

        keypoint_generation_time = round(toc - tic, 3)
    else:
        # Load camera array from working project config
        config = Configurator(working_project_dir)
        camera_array = config.get_camera_array()

    # keypoint data now exists. It looks like this:

    # sync_index,port,frame_index,frame_time,point_id,img_loc_x,img_loc_y,obj_loc_x,obj_loc_y,obj_loc_z
    # 0,1,0,0.0,4,309.9118995666504,77.12911128997803,0.0,-0.009594880365291056,0.0068349506412931586
    # 0,1,0,0.0,6,310.7944679260254,66.80278778076172,0.0,-0.023732237777556297,0.06543170253856367
    # 0,1,0,0.0,10,315.24370193481445,45.60556888580322,0.0,-0.07927442258111275,0.1532273162740576
    # and on and on...

    tracker_xy_path = raw_video_dir / f"{tracker.name}/xy_{tracker.name}.csv"

    # load in 2d data
    image_points = ImagePoints.from_csv(tracker_xy_path)

    # construct best guess of paired poses between all cameras
    tic = time.time()
    pose_network = build_paired_pose_network(image_points, camera_array)

    # initialize camera extrinsics based on best guess stereopairs
    pose_network.apply_to(camera_array)
    toc = time.time()

    pose_network_initialization_time = toc - tic

    # triangulate 2D points using initialize extrinsics
    initial_world_points = image_points.triangulate(camera_array)

    # run the bundle adjustment with these initialized values to dial in extrinsics

    tic = time.time()
    capture_volume = CaptureVolume(camera_array, initial_world_points.to_point_estimates(image_points, camera_array))
    capture_volume.optimize()
    toc = time.time()
    bundle_adjustment_time = round(toc - tic, 3)

    optimized_camera_array = capture_volume.camera_array

    # create final 3D point estimates
    tic = time.time()
    final_world_points = image_points.triangulate(optimized_camera_array)
    toc = time.time()
    triangulation_time = toc - tic

    # Save data to load into controller layer for visualization
    config.save_camera_array(optimized_camera_array)
    config.save_point_estimates(capture_volume.point_estimates)

    logger.info(
        f"""
        \nkeypoint generation: {keypoint_generation_time}\npose initialization time:{pose_network_initialization_time}
        \nbundle adj. time: {bundle_adjustment_time}\ntriangulation time {triangulation_time}
        """
    )

    ##### VISUALIZE CAPTURE VOLUME ############

    app = QApplication(sys.argv)

    controller = Controller(working_project_dir)
    controller.load_camera_array()
    controller.load_estimated_capture_volume()

    window = CaptureVolumeWidget(controller)
    # After filtering - log filtered point counts

    logger.info("Point counts loaded into Capture Volume Widget:")
    logger.info(f"  3D points (obj.shape[0]): {controller.capture_volume.point_estimates.obj.shape[0]}")
    logger.info(f"  2D observations (img.shape[0]): {controller.capture_volume.point_estimates.img.shape[0]}")
    logger.info(f"  Camera indices length: {len(controller.capture_volume.point_estimates.camera_indices)}")

    window.show()

    app.exec()
