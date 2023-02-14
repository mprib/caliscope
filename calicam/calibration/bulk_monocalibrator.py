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
    def __init__(
        self,
        config_path: Path,
        point_data_path: Path,
        calibration_sample_size: int =40,
        random_state: int = None,
    ):

        self.config = toml.load(config_path)
        self.point_data = pd.read_csv(point_data_path)

        self.ports = [int(key[4:]) for key in self.config.keys() if key[0:3] == "cam"]

        # self.corner_count_threshold = 5
        self.calibration_sample_size = calibration_sample_size
        self.random_state = (
            random_state  # if set to a number, then selections will be repeatable
        )
        self.select_calibration_points()


    def select_calibration_points(self):
        # get the mean board positions
        all_board_captures  = (self.point_data.filter(
                ["port", "sync_index", "frame_time", "point_id", "img_loc_x", "img_loc_y"]
            )
            .groupby(["port", "sync_index", "frame_time"])
            .agg({"point_id": "count", "img_loc_x": "mean", "img_loc_y": "mean"})
            .rename(
                {"img_loc_x": "x_mean", "img_loc_y": "y_mean", "point_id": "point_count"},
                axis=1,
            )
            .reset_index()
        )
        
        
        # self.board_captures = (
        #     self.point_data.filter(["port", "sync_index", "point_id"])
        #     .groupby(["port", "sync_index"])
        #     .agg({"point_id": "count"})
        #     .rename(
        #         {
        #             "point_id": "point_count",
        #         },
        #         axis=1,
        #     )
        #     .reset_index()
        # )

        # DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences
        # point_count_cutoff = 6
        point_count_cutoff = 12
        self.good_board_captures = all_board_captures[all_board_captures["point_count"]>=point_count_cutoff]

        self.randomly_selected_boards = self.good_board_captures.groupby("port").sample(
            n=self.calibration_sample_size, random_state=self.random_state, replace=True
        )


        # this is what I need to expand on...
        self.selected_boards = self.good_board_captures

        # TODO: START HERE TOMORROW, MAC: 
        # use the code below as a starting place. Get the overlap_regions
        # for the points, and use that to get the "PrimaryOverlapRegion" for the board
        # for a give port, find out how many are in each Region
        # and how many are in each port
        # Proportionally sample from the regions to achieve the desired number of boards
        
            #def get_points_with_multiport(self):
                # """
                # Pivot the port columns and assemble a new string field that will show all of the cameras that
                # observed a given corner at a single sync index.
                # """
                # points_by_multiport = (
                #     self.point_data.filter(["sync_index", "point_id", "port"])
                #     .pivot(index=["sync_index", "point_id"], columns="port", values="port")
                #     .reset_index()
                #     .fillna("")
                # )

                # def get_multiport_label(row):
                #     """
                #     returns a string of the format "_0_1_2" for points which were captured
                #     by cameras 0,1 and 2, etc...
                #     """
                #     text = ""
                #     for port in self.ports:
                #         label = row[port]
                #         if label != "":
                #             label = str(int(label))
                #             text = text + "_" + label

                #     return text

                # points_by_multiport["captured_by"] = points_by_multiport.apply(
                #     get_multiport_label, axis=1, args=()
                # )

                # return points_by_multiport

        self.calibration_points = self.point_data.merge(
            self.randomly_selected_boards, "right", ["port", "sync_index"]
        )


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
        port_monocal_data = self.calibration_points[self.calibration_points["port"]==port]

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
    # bulk_monocal.calibrate(3)

    #%%

    # get the mean board positions
    # mean_board_xy = (
    #     bulk_monocal.point_data.filter(
    #         ["port", "sync_index", "frame_time", "point_id", "img_loc_x", "img_loc_y"]
    #     )
    #     .groupby(["port", "sync_index", "frame_time"])
    #     .agg({"point_id": "count", "img_loc_x": "mean", "img_loc_y": "mean"})
    #     .rename(
    #         {"img_loc_x": "x_mean", "img_loc_y": "y_mean", "point_id": "point_count"},
    #         axis=1,
    #     )
    #     .reset_index()
    # )

    # target_board_count = 30
    # port = 3
    # point_count_cutoff = 9

    # mean_port_board_xy = mean_board_xy.loc[
    #     (mean_board_xy["port"] == port)
    #     & (mean_board_xy["point_count"] >= point_count_cutoff)
    # ].assign(use_for_calibration=False)

    # if mean_port_board_xy.shape[0] > target_board_count:
    #     pass
    # need to reduces this down even more
# %%
