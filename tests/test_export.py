# %%

from pyxy3d.trackers.holistic_tracker import HolisticTracker
from pathlib import Path
from pyxy3d import __root__
from pyxy3d.helper import copy_contents
from pyxy3d.export import xyz_to_wide_csv, xyz_to_trc

import csv
import pandas as pd


original_data_path = Path( __root__, "tests", "sessions","4_cam_recording","recording_1", "HOLISTIC")
working_data_path = Path( __root__, "tests", "sessions_copy_delete","4_cam_recording","recording_1", "HOLISTIC")

def test_export():
    copy_contents(original_data_path, working_data_path)

    tracker = HolisticTracker()
    xyz_csv_path = Path(working_data_path, f"xyz_{tracker.name}.csv")


    # this file should be created now
    xyz_labelled_path = Path(xyz_csv_path.parent, f"{xyz_csv_path.stem}_labelled.csv")
    # the file shouldn't exist yet
    assert not xyz_labelled_path.exists()
    # create it
    xyz_to_wide_csv(xyz_csv_path, tracker)
    # confirm it exists
    assert xyz_labelled_path.exists()

    # do the same with the trc file
    trc_path = Path(xyz_csv_path.parent, f"{xyz_csv_path.stem}.trc")
    assert not trc_path.exists()
    xyz_to_trc(xyz_csv_path,tracker)
    assert trc_path.exists()
    # %%

if __name__ == "__main__":
    
    test_export()