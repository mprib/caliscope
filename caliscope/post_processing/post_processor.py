import shutil
from pathlib import Path
from time import sleep

import pandas as pd

import caliscope.logger
from caliscope.cameras.camera_array import CameraArray
from caliscope.export import xyz_to_trc, xyz_to_wide_labelled
from caliscope.post_processing.gap_filling import gap_fill_xy, gap_fill_xyz
from caliscope.post_processing.smoothing import smooth_xyz
from caliscope.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.trackers.tracker_enum import TrackerEnum
from caliscope.triangulate.triangulation import triangulate_xy

logger = caliscope.logger.get(__name__)

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
        shutil.copy(Path(self.recording_path.parent.parent, "config.toml"), Path(tracker_subdirectory, "config.toml"))

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
        creates xyz_{tracker name}.csv file within the recording_path directory

        Uses the two functions above, first creating the xy points based on the tracker if they
        don't already exist, the triangulating them. Makes use of an internal method self.triangulate_xy_data

        """

        tracker_output_path = Path(self.recording_path, self.tracker_name)
        xy_csv_path = Path(tracker_output_path, f"xy_{self.tracker_name}.csv")

        # create if it doesn't already exist
        if not xy_csv_path.exists():
            self.create_xy()

        # load in 2d data and triangulate it
        logger.info("Reading in (x,y) data..")
        xy = pd.read_csv(xy_csv_path)
        if xy.shape[0] > 0:
            logger.info("Filling small gaps in (x,y) data")
            xy = gap_fill_xy(xy, max_gap_size=xy_gap_fill)
            logger.info("Beginning data triangulation")
            xyz = triangulate_xy(xy, self.camera_array)
        else:
            logger.warning("No points tracked. Terminating post-processing early.")
            return

        if xyz.shape[0] > 0:
            if APPLY_EXPERIMENTAL_POST_PROCESSING:
                logger.info("Filling small gaps in (x,y,z) data")
                xyz = gap_fill_xyz(xyz, max_gap_size=xyz_gap_fill)
                logger.info("Smoothing (x,y,z) using butterworth filter with cutoff frequency of 6hz")
                xyz = smooth_xyz(xyz, order=2, fps=self.sync_stream_manager.mean_fps, cutoff=cutoff_freq)

            logger.info("Saving (x,y,z) to csv file")
            xyz_csv_path = Path(tracker_output_path, f"xyz_{self.tracker_name}.csv")
            xyz.to_csv(xyz_csv_path)
            xyz_wide_csv_path = Path(tracker_output_path, f"xyz_{self.tracker_name}_labelled.csv")
            xyz_labelled = xyz_to_wide_labelled(xyz, self.tracker_enum.value())
            xyz_labelled.to_csv(xyz_wide_csv_path)

        else:
            logger.warning("No points triangulated. Terminating post-processing early.")
            return

        # only include trc if wanted and only if there is actually good data to export
        if include_trc and xyz.shape[0] > 0:
            trc_path = Path(tracker_output_path, f"xyz_{self.tracker_name}.trc")
            time_history_path = Path(tracker_output_path, "frame_time_history.csv")
            xyz_to_trc(
                xyz,
                tracker=self.tracker_enum.value(),
                time_history_path=time_history_path,
                target_path=trc_path,
            )


if __name__ == "__main__":
    from caliscope.controller import Controller

    workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive - The University of Texas at Austin\research\caliscope\demo")
    controller = Controller(workspace_dir)
    controller.load_camera_array()

    camera_aray = controller.camera_array
    recording_dir = Path(workspace_dir, "recordings", "STS")

    post_processor = PostProcessor(
        camera_array=camera_aray,
        recording_path=recording_dir,
        tracker_enum=TrackerEnum.HAND,
    )

    post_processor.create_xyz()
