from pathlib import Path
import pandas as pd

from pyxy3d.interface import Tracker

def xyz_to_wide_csv(xyz_path:Path, tracker:Tracker, target_path:Path):

        df_xyz = pd.read_csv(xyz_path)
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