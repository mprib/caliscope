# %%

from caliscope.trackers.holistic.holistic_tracker import HolisticTracker
from pathlib import Path
from caliscope import __root__
from caliscope.helper import copy_contents
from caliscope.export import xyz_to_wide_labelled, xyz_to_trc

import csv
import pandas as pd


original_data_path = Path( __root__, "tests", "sessions","4_cam_recording","recording_1", "HOLISTIC")
working_data_path = Path( __root__, "tests", "sessions_copy_delete","4_cam_recording","recording_1", "HOLISTIC")

def test_export():
    copy_contents(original_data_path, working_data_path)

    tracker = HolisticTracker()
    xyz_csv_path = Path(working_data_path, f"xyz_{tracker.name}.csv")
    xyz = pd.read_csv(xyz_csv_path)

    # this file should be created now
    xyz_labelled_path = Path(xyz_csv_path.parent, f"{xyz_csv_path.stem}_labelled.csv")
    # the file shouldn't exist yet
    assert not xyz_labelled_path.exists()
    # create it
    xyz_labelled = xyz_to_wide_labelled(xyz, tracker)
    xyz_labelled.to_csv(xyz_labelled_path)
    # confirm it exists
    assert xyz_labelled_path.exists()

    # do the same with the trc file
    time_history_path = Path(xyz_csv_path.parent, "frame_time_history.csv")
    trc_path = Path(xyz_csv_path.parent, f"{xyz_csv_path.stem}.trc")
    assert not trc_path.exists()

    xyz_to_trc(xyz,tracker,time_history_path, target_path=trc_path)
    assert trc_path.exists()
    # %%

if __name__ == "__main__":
    
    test_export()