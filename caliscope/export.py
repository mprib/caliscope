import caliscope.logger
from pathlib import Path
import pandas as pd
import csv

from caliscope.packets import Tracker
logger = caliscope.logger.get(__name__)

def xyz_to_wide_labelled(xyz:pd.DataFrame, tracker:Tracker)->pd.DataFrame:
    """
    Will save a csv file in the same directory as the long xyz point data
    Column headings will be based on the point_id names in the Tracker
    """

    # save out named data in a tabular format
    xyz = xyz.rename(
        {
            "x_coord": "x",
            "y_coord": "y",
            "z_coord": "z",
        },
        axis=1,
    )
    xyz = xyz[["sync_index", "point_id", "x", "y", "z"]]

    xyz["point_name"] = xyz["point_id"].map(tracker.get_point_name)
    # pivot the DataFrame wider
    df_wide = xyz.pivot_table(
        index=["sync_index"], columns="point_name", values=["x", "y", "z"]
    )
    # flatten the column names
    df_wide.columns = ["{}_{}".format(y, x) for x, y in df_wide.columns]
    # reset the index
    df_wide = df_wide.reset_index()
    # merge the rows with the same sync_index
    df_merged = df_wide.groupby("sync_index").agg("first")
    # sort the dataframe
    df_merged = df_merged.sort_index(axis=1, ascending=True)
    return df_merged

def xyz_to_trc(xyz:pd.DataFrame, tracker:Tracker, time_history_path:Path, target_path:Path):
    """
    Will save a .trc file in the same folder as the long xyz data
    relies on xyz_to_wide_csv for input data
    """  
    # create xyz_labelled file to provide input for trc creation
    xyz_labelled = xyz_to_wide_labelled(xyz, tracker)

    # from here I need to get a .trc file format. For part of that I also need to know the framerate.
    # time_history_path = Path(target_path.parent, "frame_time_history.csv")
    time_history = pd.read_csv(time_history_path)

    # get the mean time by sync index
    # Group by 'sync_index' and calculate mean 'frame_time'
    sync_time = time_history.groupby("sync_index")["frame_time"].mean()

    # Shift times so that it starts at zero
    min_time = sync_time.min()
    sync_time = round(sync_time - min_time,3)
    xyz_labelled.insert(1, "mean_frame_time", sync_time)
    # %%

    # Calculate time differences between consecutive frames
    xyz_labelled.sort_values(by="mean_frame_time", inplace=True)
    xyz_labelled["time_diff"] = xyz_labelled["mean_frame_time"].diff()

        

    # Calculate frame rate for each pair of frames (avoid division by zero)
    xyz_labelled["frame_rate"] = xyz_labelled["time_diff"].apply(
        lambda x: 1 / x if x != 0 else 0
    )

    # Calculate mean frame rate (drop the first value which is NaN due to the diff operation)
    mean_frame_rate = xyz_labelled["frame_rate"].dropna().mean()
    
    # Rename 'sync_index' to 'Frame' and 'mean_frame_time' to 'Time'
    xyz_labelled = xyz_labelled.reset_index() # need sync_index to be just a regular column
    xyz_labelled.rename(columns={'sync_index': 'Frame', 'mean_frame_time': 'Time'}, inplace=True)
    xyz_labelled.drop(columns=["time_diff", "frame_rate"], inplace=True) # no longer needed

    # Make sure all following fields are in alphabetical order
    # First, get the columns to be sorted, i.e., all columns excluding 'Frame' and 'Time'
    cols_to_sort = xyz_labelled.columns.tolist()[2:]
    cols_to_sort = [col for col in cols_to_sort if not col.startswith("face")]

    # Now, sort these columns
    cols_to_sort.sort()
    # Now, create the final column order and rearrange the DataFrame
    final_col_order = ['Frame', 'Time'] + cols_to_sort
    xyz_labelled = xyz_labelled[final_col_order]

    # trying a fix...
    xyz_labelled["Frame"] = xyz_labelled["Frame"].astype(int)
    
    # Get column names from dataframe
    columns = xyz_labelled.columns

    # Remove '_x', '_y', and '_z' suffixes and get unique names
    tracked_points = list(
        set([col.rsplit("_", 1)[0] for col in columns if col.endswith(("_x", "_y", "_z"))])
    )
    tracked_points.sort()


    num_markers = len(tracked_points)
    data_rate = int(mean_frame_rate)
    units = "m"
    original_data_rate = int(mean_frame_rate)
    orig_data_start_frame = 0
    num_frames = len(xyz_labelled)-1

    trc_path = Path(target_path.parent,f"{target_path.stem}.trc")
    trc_filename = str(trc_path)

    # this will create the formatted .trc file
    with open(trc_path, 'wt', newline='', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["PathFileType",
                            "4", 
                            "(X/Y/Z)",	
                            trc_filename])
        tsv_writer.writerow(["DataRate",
                            "CameraRate",
                            "NumFrames",
                            "NumMarkers", 
                            "Units",
                            "OrigDataRate",
                            "OrigDataStartFrame",
                            "OrigNumFrames"])
        tsv_writer.writerow([data_rate, 
                            int(mean_frame_rate),
                            num_frames, 
                            num_markers, 
                            units, 
                            int(mean_frame_rate), 
                            orig_data_start_frame, 
                            num_frames])

        # create names of trajectories, skipping two columns (top of table)
        header_names = ['Frame#', 'Time']
        for trajectory in tracked_points:
            header_names.append(trajectory)
            header_names.append("")
            header_names.append("")    

        tsv_writer.writerow(header_names)

        # create labels for x,y,z axes (below landmark names in header)
        header_names = ["",""]
        for i in range(1,len(tracked_points)+1):
            header_names.append("X"+str(i))
            header_names.append("Y"+str(i))
            header_names.append("Z"+str(i))

        tsv_writer.writerow(header_names)

        # the .trc fileformat expects a blank fourth line
        tsv_writer.writerow("")


        # this is here primarily for testing purposes right now..
        # filLs None with zeros.
        # df_xyz_labelled.fillna(0,inplace=True)


        # and finally actually write the trajectories
        for row in range(0, len(xyz_labelled)):
            row_data = xyz_labelled.iloc[row].tolist()
    
            # Convert the 'Frame' column value to int to satisfy trc format requirements
            frame_index = xyz_labelled.columns.get_loc('Frame')
            row_data[frame_index] = int(row_data[frame_index])

            tsv_writer.writerow(row_data)