# caliscope/post_processing/post_processor.py

import logging
import shutil
from pathlib import Path
from time import sleep

from caliscope.cameras.camera_array import CameraArray
from caliscope.export import xyz_to_trc, xyz_to_wide_labelled
from caliscope.core.point_data import ImagePoints
from caliscope.managers.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.trackers.tracker_enum import TrackerEnum

logger = logging.getLogger(__name__)

# gap filling and filtering is outside the current scope of the project so I'm toggling this off for now
APPLY_EXPERIMENTAL_POST_PROCESSING = False


class PostProcessor:
    """
    The post processer operates independently of the session. It does not need to worry about camera management.
    Provide it with a path to the directory that contains the following:
    - config.toml
    - frame_time.csv
    - .mp4 files

    The post processor will archive the active config.toml file into the subdirectory
    """

    def __init__(self, camera_array: CameraArray, recording_path: Path, tracker_enum: TrackerEnum):
        self.camera_array = camera_array
        self.recording_path = recording_path
        self.tracker_enum = tracker_enum
        self.tracker_name = tracker_enum.name
        self.tracker = tracker_enum.value()

        # save out current camera array to output folder
        tracker_subdirectory = Path(self.recording_path, self.tracker_name)
        tracker_subdirectory.mkdir(exist_ok=True, parents=True)
        shutil.copy(
            Path(self.recording_path.parent.parent, "camera_array.toml"),
            Path(tracker_subdirectory, "camera_array.toml"),
        )

        logger.info(f"Creating sync stream manager for videos stored in {self.recording_path}")
        self.sync_stream_manager = SynchronizedStreamManager(
            self.recording_path, self.camera_array.cameras, self.tracker
        )

    def create_xy(self, fps_target=100, include_video=True):
        """
        Reads through all .mp4  files in the recording path and applies the tracker to them
        The xy_TrackerName.csv file is saved out to the same directory by the VideoRecorder

        Note that high fps target and including video will increase processing overhead
        """
        self.sync_stream_manager.process_streams(include_video=include_video, fps_target=fps_target)

        while self.sync_stream_manager.recorder.recording:
            sleep(1)
            percent_complete = int(
                (self.sync_stream_manager.recorder.sync_index / self.sync_stream_manager.mean_frame_count) * 100
            )
            logger.info(f"(Stage 1 of 2): {percent_complete}% of frames processed for (x,y) landmark detection")

    def create_xyz(self, xy_gap_fill=3, xyz_gap_fill=3, cutoff_freq=6, include_trc=True) -> None:
        """
        Creates xyz_{tracker name}.csv file within the recording_path directory.

        This method orchestrates the post-processing pipeline using validated
        data objects (ImagePoints, WorldPoints) for a clear and robust workflow.
        """
        tracker_output_path = Path(self.recording_path, self.tracker_name)
        xy_csv_path = Path(tracker_output_path, f"xy_{self.tracker_name}.csv")

        if not xy_csv_path.exists():
            logger.info(f"{xy_csv_path} not found. Running landmark detection to create it.")
            self.create_xy()

        logger.info(f"Loading (x,y) data from {xy_csv_path}...")
        try:
            xy_data = ImagePoints.from_csv(xy_csv_path)
        except Exception as e:
            logger.error(f"Could not load or validate data from {xy_csv_path}. Error: {e}")
            return

        if xy_data.df.empty:
            logger.warning("No points found in xy data file. Terminating post-processing early.")
            return

        # Start the processing pipeline
        logger.info(f"Filling small gaps in (x,y) data (max_gap={xy_gap_fill})...")
        filled_xy = xy_data.fill_gaps(max_gap_size=xy_gap_fill)

        logger.info("Beginning data triangulation...")
        xyz_data = filled_xy.triangulate(self.camera_array)

        if xyz_data.df.empty:
            logger.warning("No points were triangulated. Terminating post-processing early.")
            return

        # This variable will hold the final state of the data
        final_xyz_data = xyz_data

        if APPLY_EXPERIMENTAL_POST_PROCESSING:
            logger.info(f"Filling small gaps in (x,y,z) data (max_gap={xyz_gap_fill})...")
            final_xyz_data = final_xyz_data.fill_gaps(max_gap_size=xyz_gap_fill)

            logger.info(f"Smoothing (x,y,z) using Butterworth filter (cutoff_freq={cutoff_freq}Hz)...")
            final_xyz_data = final_xyz_data.smooth(fps=self.sync_stream_manager.mean_fps, cutoff_freq=cutoff_freq)

        logger.info("Saving (x,y,z) data to CSV files...")
        xyz_csv_path = Path(tracker_output_path, f"xyz_{self.tracker_name}.csv")
        final_xyz_data.df.to_csv(xyz_csv_path, index=False)

        xyz_wide_csv_path = Path(tracker_output_path, f"xyz_{self.tracker_name}_labelled.csv")
        xyz_labelled = xyz_to_wide_labelled(final_xyz_data.df, self.tracker_enum.value())
        xyz_labelled.to_csv(xyz_wide_csv_path, index=False)

        if include_trc:
            trc_path = Path(tracker_output_path, f"xyz_{self.tracker_name}.trc")
            time_history_path = Path(self.recording_path, "frame_time_history.csv")
            xyz_to_trc(
                final_xyz_data.df,
                tracker=self.tracker_enum.value(),
                time_history_path=time_history_path,
                target_path=trc_path,
            )
