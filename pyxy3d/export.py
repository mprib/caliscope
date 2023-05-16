from pathlib import Path
import pandas as pd
import csv

from pyxy3d.interface import Tracker

def xyz_to_wide_csv(xyz_path:Path, tracker:Tracker):
    """
    Will save a csv file in the same directory as the long xyz point data
    Column headings will be based on the point_id names in the Tracker
    """
    df_xyz = pd.read_csv(xyz_path)
    target_path = Path(xyz_path.parent, f"{xyz_path.stem}_labelled.csv")
    # save out named data in a tabular format
    df_xyz = df_xyz.rename(
        {
            "x_coord": "x",
            "y_coord": "y",
            "z_coord": "z",
        },
        axis=1,
    )
    df_xyz = df_xyz[["sync_index", "point_id", "x", "y", "z"]]

    df_xyz["point_name"] = df_xyz["point_id"].map(tracker.get_point_name)
    # pivot the DataFrame wider
    df_wide = df_xyz.pivot_table(
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
    df_merged.to_csv(target_path)

def xyz_to_trc(xyz_path:Path, tracker:Tracker):
    """
    Will save a .trc file in the same folder as the long xyz data
    relies on xyz_to_wide_csv for input data
    """  

    xyz_labelled_path = Path(xyz_path.parent, f"{xyz_path.stem}_labelled.csv")

    # load in the data and confirm it's populated
    df_xyz_labelled = pd.read_csv(xyz_labelled_path)
    # assert(not df_xyz_labelled.empty)

    # from here I need to get a .trc file format. For part of that I also need to know the framerate.
    time_history_path = Path(xyz_path.parent, "frame_time_history.csv")
    time_history = pd.read_csv(time_history_path)

    # get the mean time by sync index
    # Group by 'sync_index' and calculate mean 'frame_time'
    sync_time = time_history.groupby("sync_index")["frame_time"].mean()

    # Shift times so that it starts at zero
    min_time = sync_time.min()
    sync_time = round(sync_time - min_time,3)
    df_xyz_labelled.insert(1, "mean_frame_time", sync_time)
    # %%

    # Calculate time differences between consecutive frames
    df_xyz_labelled.sort_values(by="mean_frame_time", inplace=True)
    df_xyz_labelled["time_diff"] = df_xyz_labelled["mean_frame_time"].diff()

    # Calculate frame rate for each pair of frames (avoid division by zero)
    df_xyz_labelled["frame_rate"] = df_xyz_labelled["time_diff"].apply(
        lambda x: 1 / x if x != 0 else 0
    )

    # Calculate mean frame rate (drop the first value which is NaN due to the diff operation)
    mean_frame_rate = df_xyz_labelled["frame_rate"].dropna().mean()

    # Get column names from dataframe
    columns = df_xyz_labelled.columns

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
    num_frames = len(df_xyz_labelled)-1

    trc_path = Path(xyz_path.parent,f"{xyz_path.stem}.trc")
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

        # Rename 'sync_index' to 'Frame' and 'mean_frame_time' to 'Time'
        df_xyz_labelled.rename(columns={'sync_index': 'Frame', 'mean_frame_time': 'Time'}, inplace=True)
        df_xyz_labelled.drop(columns=["time_diff", "frame_rate"], inplace=True) # no longer needed
        
        # Make sure all following fields are in alphabetical order
        # First, get the columns to be sorted, i.e., all columns excluding 'Frame' and 'Time'
        cols_to_sort = df_xyz_labelled.columns.tolist()[2:]

        # Now, sort these columns
        cols_to_sort.sort()

        # Now, create the final column order and rearrange the DataFrame
        final_col_order = ['Frame', 'Time'] + cols_to_sort
        df_xyz_labelled = df_xyz_labelled[final_col_order]
    
        # and finally actually write the trajectories
        for row in range(0, len(df_xyz_labelled)):
            tsv_writer.writerow(df_xyz_labelled.iloc[row].tolist())


