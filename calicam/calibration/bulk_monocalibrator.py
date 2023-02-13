#%%

import calicam.logger

logger = calicam.logger.get(__name__)

import cv2
import sys
import pandas as pd
from calicam import __root__

sys.path.insert(0, __root__)
from pathlib import Path
import numpy as np
import toml


class BulkMonocalibrator:
    def __init__(self, config_path: Path, point_data_path: Path):

        self.config = toml.load(config_path)
        self.point_data = pd.read_csv(point_data_path)

        self.ports = [int(key[4:]) for key in self.config.keys() if key[0:3] == "cam"]

        self.corner_count_threshold = 11
        self.top_x_count = 9

        self.points_with_multiport = self.get_points_with_multiport()

    def get_points_with_multiport(self):
        """
        Pivot the port columns and assemble a new string field that will show all of the cameras that
        observed a given corner at a single sync index.
        """
        points_by_multiport = (
            self.point_data.filter(["sync_index", "point_id", "port"])
            .pivot(index=["sync_index", "point_id"], columns="port", values="port")
            .reset_index()
            .fillna("")
        )

        def get_multiport_label(row):
            """
            returns a string of the format "_0_1_2" for points which were captured
            by cameras 0,1 and 2, etc...
            """
            text = ""
            for port in self.ports:
                label = row[port]
                if label != "":
                    label = str(int(label))
                    text = text + "_" + label

            return text

        points_by_multiport["captured_by"] = points_by_multiport.apply(
            get_multiport_label, axis=1, args=()
        )

        return points_by_multiport

    def get_port_points(self, port):
        #%%
        # self = bulk_monocal
        # port = 2
        
        #%%
        single_port_points = self.points_with_multiport.loc[
            self.points_with_multiport[port] == port
        ].assign(port=port)

        board_counts = (
            single_port_points.filter(["sync_index", "point_id"])
            .groupby("sync_index")
            .count()
            .rename({"point_id": "corner_count"}, axis=1)
        )

        board_seen_by = (
            single_port_points.groupby(["port", "sync_index", "captured_by"])
            .agg("count")
            .rename({"point_id": "seen_by_count"}, axis=1)
            .reset_index()
        )

        board_most_seen_by = (
            board_seen_by.groupby(["port", "sync_index"])
            .first()
            .drop(
                columns="seen_by_count"
            )  # this no longer means much...only for one "seenby group"
            .rename({"captured_by": "most_captured_by"}, axis=1)
            .reset_index()
        )

        board_counts_most_seen_by = board_counts.merge(
            board_most_seen_by, "left", on=["sync_index"]
        )

        criteria = (
            board_counts_most_seen_by["corner_count"] >= self.corner_count_threshold
        )

        board_counts_most_seen_by = board_counts_most_seen_by[criteria]
        board_counts_most_seen_by = (
            board_counts_most_seen_by
            #  .reset_index()
            .groupby("most_captured_by")
            .head(self.top_x_count)
            .reset_index()
            .sort_values(["most_captured_by"])
        )

        port_monocal_data = self.point_data.merge(
            board_counts_most_seen_by, "right", ["sync_index", "port"]
        )
        #%%

        return port_monocal_data
        
    def calibrate(self, port):

        """
        port_monocal_data: a DataFrame that is a curated flat-file version of the
            point_data.csv file. This contains only data for one camera port, and
            only a subset of the boards are represented.

            This subset is determined previously by the
        Use the recorded image corner positions along with the objective
        corner positions based on the board definition to calculated
        the camera matrix and distortion parameters
        """
        port_monocal_data = self.get_port_points(port)

        resolution = self.config["cam_" + str(port)]["resolution"]

        sync_indices = port_monocal_data["sync_index"].to_numpy().round().astype(int)
        img_loc_x = port_monocal_data["img_loc_x"].to_numpy().astype(np.float32)
        img_loc_y = port_monocal_data["img_loc_y"].to_numpy().astype(np.float32)
        board_loc_x = port_monocal_data["board_loc_x"].to_numpy().astype(np.float32)
        board_loc_y = port_monocal_data["board_loc_y"].to_numpy().astype(np.float32)
        board_loc_z = board_loc_x * 0  # all on the same plane

        # build the actual inputs for the calibration...
        img_x_y = np.vstack([img_loc_x, img_loc_y]).T
        board_x_y_z = np.vstack([board_loc_x, board_loc_y, board_loc_z]).T

        import time

        print(time.time())
        img_locs = []  #  np.ndarray([])
        board_locs = []  # np.ndarray([])
        for sync_index in np.unique(sync_indices):
            same_frame = sync_indices == sync_index
            # np.hstack([img_locs, img_x_y[same_frame]])
            # np.hstack([board_locs, board_x_y_z[same_frame]])
            img_locs.append(img_x_y[same_frame])
            board_locs.append(board_x_y_z[same_frame])

        print(time.time())
        print(f"Using {len(img_locs)} board captures to calibrate camera....")

        start = time.time()
        logger.info(f"Calibrating camera {port}....")
        error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            board_locs, img_locs, resolution, None, None
        )
        elapsed = time.time() - start

        print(f"{elapsed} seconds elapsed to perform one camera calibration")
        logger.info(f"Error: {error}")
        logger.info(f"Camera Matrix: {mtx}")
        logger.info(f"Distortion: {dist}")



#%%
if __name__ == "__main__":
    #%%
    from pathlib import Path
    
    # set inputs
    session_path = Path(__root__, "tests", "5_cameras")

    config_path = Path(session_path, "config.toml")
    point_data_path = Path(session_path, "recording", "point_data.csv")

    bulk_monocal = BulkMonocalibrator(config_path, point_data_path)
    bulk_monocal.calibrate(3)

# %%
