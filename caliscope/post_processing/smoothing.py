
import caliscope.logger


from caliscope import __root__
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
logger = caliscope.logger.get(__name__)


# Define Butterworth filter functions
def butter_lowpass(cutoff, fs, order=2):
    # cutoff frequency must be less than half the frame rate
    if cutoff<fs/2:
        logger.warn("Low pass filter failing due to excessively low fps / excessively high filter cutoff.")


    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fps, order=2):
    b, a = butter_lowpass(cutoff, fps, order=order)

    # need to adjust for short input sequences
    padlen = min(len(data) - 1, 3 * (max(len(a), len(b)) - 1))
    y = filtfilt(b, a, data, padlen=padlen)
    return y

def _smooth(landmark_data:pd.DataFrame, order, fps, cutoff, coord_names, index_name)->pd.DataFrame:

    shifted_index_name = f"{index_name}_shifted"

    # get continuous groups of a given point that can be filtered
    landmark_data = landmark_data.sort_values(by=["point_id", index_name])
    landmark_data[shifted_index_name] = landmark_data[index_name].shift(1)
    landmark_data["new_smooth_group"] = (
        landmark_data[index_name] != landmark_data[shifted_index_name] + 1
    )
    landmark_data["smooth_group_index"] = landmark_data["new_smooth_group"].cumsum()
    landmark_data = landmark_data.drop([shifted_index_name, "new_smooth_group"], axis=1)


    # Apply the filter to each piecewise group
    logger.info("Applying butterworth filter to point coordinates")
    for coord in coord_names:
        landmark_data[coord] = landmark_data.groupby(["smooth_group_index"])[
            coord
        ].transform(butter_lowpass_filter, cutoff, fps, order)
    
    landmark_data = landmark_data.sort_values([index_name, "point_id"])
    
    return landmark_data   



def _smooth_xy(xy: pd.DataFrame, order, fps, cutoff)->pd.DataFrame:
    """
    Running this function takes a considerable amount of time. Dramatically more than 
    the xyz version. So just let this be for now and don't bother implementing it
    in practice.
    
    Gap fill the xy, but don't bother with the smoothing for now.
    
    """
    # note that in future refactors, the xy coordinates my only have a frame_index and not a sync_index
    index_name = "sync_index" 
    coord_names = ["img_loc_x","img_loc_y"]
    
    return _smooth(xy, order, fps, cutoff, coord_names,index_name)

def smooth_xyz(xyz: pd.DataFrame, order, fps, cutoff)->pd.DataFrame:
    index_name = "sync_index"
    coord_names = ["x_coord", "y_coord", "z_coord"]
    
    return _smooth(xyz, order, fps, cutoff, coord_names,index_name)