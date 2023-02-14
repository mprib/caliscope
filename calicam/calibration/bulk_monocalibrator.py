import calicam.logger

logger = calicam.logger.get(__name__)

import cv2
import sys
import pandas as pd
from calicam import __root__

import time

sys.path.insert(0, __root__)
from pathlib import Path
import numpy as np
import toml
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor, as_completed

class BulkMonocalibrator:
    def __init__(
        self,
        config_path: Path,
        point_data_path: Path,
        calibration_sample_size: int = 40,
        random_state: int = None,
    ):
        self.config_path = config_path
        self.config = toml.load(config_path)
        self.ports = [int(key[4:]) for key in self.config.keys() if key[0:3] == "cam"]
        self.calibration_sample_size = calibration_sample_size

        # import point data, adding coverage regions to each port
        raw_point_data = pd.read_csv(point_data_path)
        self.all_point_data = self.points_with_coverage_region(raw_point_data)
        self.all_boards = self.get_boards_with_coverage()

        # if set to a number, then selections will be repeatable. primarily used for testing
        self.random_state = random_state

    def get_boards_with_coverage(self):
        """
        create a dataframe of all the boards, including the most prevalent
        coverage region for each board
        Begin by getting the total number of corners observed for each board
        splitting out into its own creation method just for readability during init
        """

        board_points = (
            self.all_point_data.filter(["port", "sync_index", "point_id"])
            .groupby(["port", "sync_index"])
            .agg({"point_id": "count"})
            .rename(columns={"point_id": "point_count"})
            .reset_index()
        )

        # find the primary region each board is in, then merge in total counts
        all_boards = (
            self.all_point_data.filter(
                ["port", "sync_index", "point_id", "coverage_region"]
            )
            .groupby(["port", "sync_index", "coverage_region"])
            .agg({"point_id": "count"})
            .reset_index()
            .rename(
                {"point_id": "point_count"},
                axis=1,
            )
            .sort_values(
                ["port", "sync_index", "point_count"],
                axis=0,
                ascending=[True, True, False],
            )
            .groupby(["port", "sync_index"])
            .first()
            .reset_index()
            .rename(columns={"coverage_region": "primary_coverage_region"})
            .drop("point_count", axis=1)  # no longer means anthing
            .merge(
                board_points, "left", ["port", "sync_index"]
            )  # merge in a point_count that means something
        )

        return all_boards

    def get_calibration_points(self, port: int):
        """
        Provides a curated dataframe of point data. The overall point data is initially
        restricted to only those boards that have at least 6 points. This remaining data
        is grouped according to the Primary Coverage Region, which should hopefully provide
        more breadth of coverage. The random selection is weighted to strongly favor boards that
        have more points in view.
        """

        # DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences
        point_count_cutoff = 6

        port_boards = self.all_boards[self.all_boards["port"] == port]

        good_board_captures = port_boards[
            port_boards["point_count"] >= point_count_cutoff
        ]

        board_count = good_board_captures.shape[0]
        sampled_proportion = min(self.calibration_sample_size / board_count, 1)

        sampling_weights = good_board_captures["point_count"] ** 3

        randomly_selected_boards = good_board_captures.groupby(
            "primary_coverage_region"
        ).sample(
            frac=sampled_proportion,
            weights=sampling_weights,
            random_state=self.random_state,
            replace=False,
        )

        calibration_points = self.all_point_data.merge(
            randomly_selected_boards, "right", ["port", "sync_index"]
        )

        return calibration_points

    def points_with_coverage_region(self, point_data: pd.DataFrame):
        """
        Pivot the port columns and assemble a new string field that will show all of the cameras that
        observed a given corner at a single sync index.
        """

        points_w_pivoted_ports = (
            point_data.filter(["sync_index", "point_id", "port"])
            .pivot(index=["sync_index", "point_id"], columns="port", values="port")
            .reset_index()
            .fillna("")
        )

        def get_coverage_region(row, ports):
            """
            returns a string of the format "_0_1_2" for points which were captured
            by cameras 0,1 and 2, etc...
            """
            text = ""
            for port in ports:
                label = row[port]
                if label != "":
                    label = str(int(label))
                    text = text + "_" + label

            return text

        points_w_pivoted_ports["coverage_region"] = points_w_pivoted_ports.apply(
            get_coverage_region, axis=1, args=(self.ports,)
        )

        points_w_pivoted_ports = points_w_pivoted_ports.filter(
            ["sync_index", "point_id", "coverage_region"]
        )
        points_w_regions = point_data.merge(
            points_w_pivoted_ports, "left", ["sync_index", "point_id"]
        )

        return points_w_regions

    def calibrate(self, port):
        port_monocal_data = self.get_calibration_points(port)

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

        # print(time.time())
        img_locs = []
        board_locs = []
        for sync_index in np.unique(sync_indices):
            same_frame = sync_indices == sync_index
            img_locs.append(img_x_y[same_frame])
            board_locs.append(board_x_y_z[same_frame])

        grid_count = len(img_locs)
        # print(time.time())
        logger.info(
            f"Using {grid_count} board captures to calibrate camera {port}..."
        )

        start = time.time()
        logger.info(f"Calibrating camera {port}....")
        error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            board_locs, img_locs, resolution, None, None
        )
        elapsed = time.time() - start

        logger.info(
            f"{round(elapsed,2)} seconds elapsed to perform calibration of camera at port {port}"
        )
        logger.info(f"Camera {port} Error: {error}")
        
        return port, error, mtx, dist, grid_count

    def calibrate_all(self, parallel=True):

        if parallel:
            start = time.time()
            
            with ProcessPoolExecutor() as executor:
                processes = [executor.submit(self.calibrate, port) for port in self.ports]
                
                for p in as_completed(processes):
                    port, error, mtx, dist, grid_count = p.result()
                
                    self.config["cam_"+str(port)]["error"] = error
                    self.config["cam_"+str(port)]["camera_matrix"] = mtx
                    self.config["cam_"+str(port)]["distortion"] = dist
                    self.config["cam_"+str(port)]["grid_count"] = grid_count   

            elapsed = time.time() - start
            logger.info(
                f"Total time to calibrate all ports in parallel is {round(elapsed, 2)} seconds"
            )

        if not parallel:
            start = time.time()
            for port in self.ports:
                _, error, mtx, dist, grid_count = self.calibrate(port)
                
                self.config["cam_"+str(port)]["error"] = error
                self.config["cam_"+str(port)]["camera_matrix"] = mtx
                self.config["cam_"+str(port)]["distortion"] = dist
                self.config["cam_"+str(port)]["grid_count"] = grid_count   
                self.calibrate(port)

            elapsed = time.time() - start
            logger.info(
                f"Total time to calibrate all ports synchronously is {round(elapsed, 2)} seconds"
            )
         
        with open(self.config_path, "w") as f:
            toml.dump(self.config, f)


#%%
if __name__ == "__main__":
    #%%
    from pathlib import Path

    # set inputs
    session_path = Path(__root__, "tests", "5_cameras")

    config_path = Path(session_path, "config.toml")
    point_data_path = Path(session_path, "recording", "point_data.csv")

    bulk_monocal = BulkMonocalibrator(
        config_path, point_data_path, calibration_sample_size=30
    )

    # bulk_monocal.calibrate_all(parallel=False)
    bulk_monocal.calibrate_all()
# %%
