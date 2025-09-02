import sys
from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import QApplication
from scipy.stats import pearsonr

import caliscope.logger
from caliscope import __root__
from caliscope.configurator import Configurator
from caliscope.gui.vizualize.playback_triangulation_widget import PlaybackTriangulationWidget

# specify a source directory (with recordings)
from caliscope.helper import copy_contents
from caliscope.post_processing.smoothing import smooth_xyz

logger = caliscope.logger.get(__name__)
original_base_data_directory = Path(__root__, "tests", "reference", "base_data")
base_data_directory = Path(original_base_data_directory.parent.parent, "reference_delete", "base_data")
copy_contents(original_base_data_directory, base_data_directory)

xyz_path = Path(base_data_directory, "SIMPLE_HOLISTIC", "xyz_SIMPLE_HOLISTIC.csv")
config = Configurator(base_data_directory)


def test_smoothing_xyz():
    # Define your filter parameters
    order = 2
    # fs = config.get_fps_recording()  # sample rate, Hz
    fs = 15  # the value originally stored in the config file
    # note that the cutoff must be < 0.5*(sampling rate, a.k.a. nyquist frequency)
    cutoff = 6  # desired cutoff frequency, Hz

    xyz = pd.read_csv(xyz_path)
    xyz_smoothed = smooth_xyz(xyz, order, fs, cutoff)

    # save out the filterd data
    destination_path = Path(xyz_path.parent, xyz_path.stem + "_filtered.csv")
    logger.info(f"Saving filtered data to {destination_path}")
    xyz_smoothed.to_csv(destination_path)

    # Assertion 3: Value range (we'll check this for each of the coordinate columns)
    for coord in ["x_coord", "y_coord", "z_coord"]:
        assert (
            xyz_smoothed[coord].std() <= xyz[coord].std()
        ), f"The standard deviation of the smoothed {coord} data should be less than the original data."

    # Assertion 4: Preservation of trends (again, we'll check for each coordinate)
    for coord in ["x_coord", "y_coord", "z_coord"]:
        correlation, _ = pearsonr(xyz[coord], xyz_smoothed[coord])
        logger.info(f"The correlation for {coord} is {correlation}")
        assert (
            correlation > 0.9
        ), f"The correlation between the original and smoothed {coord} data should be close to 1."


if __name__ == "__main__":
    test_smoothing_xyz()

    app = QApplication(sys.argv)

    # load in the data for the playback
    camera_array = config.get_camera_array()
    filtered_data_path = Path(base_data_directory, "SIMPLE_HOLISTIC", "xyz_SIMPLE_HOLISTIC_filtered.csv")
    filtered_data = pd.read_csv(filtered_data_path)

    # create and show the playback widget
    filtered_window = PlaybackTriangulationWidget(camera_array)
    filtered_window.set_xyz(filtered_data)
    filtered_window.show()

    app.exec()
