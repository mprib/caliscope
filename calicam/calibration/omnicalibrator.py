#%%
import sys
from pathlib import Path
# sys.path.insert(0,Path(__file__).parent.parent.parent)


import calicam.logger
logger = calicam.logger.get(__name__)

import cv2
import pandas as pd
from calicam import __root__

import time

sys.path.insert(0, __root__)
import numpy as np
import toml
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations


class OmniCalibrator:
    def __init__(
        self,
        config_path: Path,
        point_data_path: Path,
    ):
        self.config_path = config_path
        self.config = toml.load(config_path)
        self.ports = [int(key[4:]) for key in self.config.keys() if key[0:3] == "cam"]
        self.pairs = [(i, j) for i, j in combinations(self.ports, 2) if i < j]

        # import point data, adding coverage regions to each port
        raw_point_data = pd.read_csv(point_data_path)
        self.all_point_data = self.points_with_coverage_region(raw_point_data)
        self.all_boards = self.get_boards_with_coverage()

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

            text = text + "_"

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


    ########################################## BEGIN MONOCALIBRATION SPECIFIC CODE #######################################################
    def get_monocalibration_points(self, port: int, sample_size: int, random_state:int):
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
        sampled_proportion = min(sample_size / board_count, 1)

        sampling_weights = good_board_captures["point_count"] ** 3

        randomly_selected_boards = good_board_captures.groupby(
            "primary_coverage_region"
        ).sample(
            frac=sampled_proportion,
            weights=sampling_weights,
            random_state=random_state,
            replace=False,
        )

        calibration_points = self.all_point_data.merge(
            randomly_selected_boards, "right", ["port", "sync_index"]
        )

        return calibration_points


    def monocalibrate(self, port, sample_size, random_state):
        # NOTE: This data cleanup can be refactored to take advantage of the cal input method
        # currently used with the stereocal method. Just not really a priority right now given
        # that I'm not terribly happy with this monocalibration outcome anyways...
        port_monocal_data = self.get_monocalibration_points(port)

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
        logger.info(f"Using {grid_count} board captures to calibrate camera {port}...")

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

    def monocalibrate_all(self, sample_size=20, random_state=1, parallel=True):
        """
        NOTE: This will run and provide calibration outputs, however it appears that these come with higher errors
        and more bizarre fits than you get when calibrating each camera individualy. I suspect that the central
        challenge is providing the most appropriate boards to the calibrator. Future iterations of this may seek to
        select boards that likely have foreshortening (more variation in connected corner length?) and that come
        from a broader swath of the frame. The random selections here may just not be enough to get the job done.
        """
        if parallel:
            start = time.time()

            with ProcessPoolExecutor() as executor:
                processes = [
                    executor.submit(self.monocalibrate, port, sample_size,random_state) for port in self.ports
                ]

                for p in as_completed(processes):
                    port, error, mtx, dist, grid_count = p.result()

                    self.config["cam_" + str(port)]["error"] = error
                    self.config["cam_" + str(port)]["camera_matrix"] = mtx
                    self.config["cam_" + str(port)]["distortion"] = dist
                    self.config["cam_" + str(port)]["grid_count"] = grid_count

            elapsed = time.time() - start
            logger.info(
                f"Total time to calibrate all ports in parallel is {round(elapsed, 2)} seconds"
            )

        if not parallel:
            start = time.time()
            for port in self.ports:
                _, error, mtx, dist, grid_count = self.monocalibrate(port,sample_size, random_state)

                self.config["cam_" + str(port)]["error"] = error
                self.config["cam_" + str(port)]["camera_matrix"] = mtx
                self.config["cam_" + str(port)]["distortion"] = dist
                self.config["cam_" + str(port)]["grid_count"] = grid_count
                self.monocalibrate(port)

            elapsed = time.time() - start
            logger.info(
                f"Total time to calibrate all ports synchronously is {round(elapsed, 2)} seconds"
            )

        with open(self.config_path, "w") as f:
            toml.dump(self.config, f)

    ##################################### BEGIN STEREOCALIBRATION SPECIFIC CODE ################################################
    
    def get_stereopair_data(self, pair, boards_sampled, random_state=1):
        
        # convenience function to get the points that are in the overlap regions of the pairs
        def in_pair(row, pair):

            a, b = pair
            port_check = row.port == a or row.port == b

            a, b = str(a), str(b)

            region_check = ("_" + a + "_") in row.coverage_region and (
                "_" + b + "_"
            ) in row.coverage_region
            return region_check and port_check

        # flag the points that belong to the pair overlap regions
        self.all_point_data["in_pair"] = self.all_point_data.apply(in_pair, axis=1, args=(pair,))

        # group points into boards and get the total count for sample weighting below
        pair_points = self.all_point_data[self.all_point_data["in_pair"] == True]
        pair_boards = (
            pair_points.filter(["sync_index", "port", "point_id"])
            .groupby(["sync_index", "port"])
            .agg("count")
            .rename({"point_id": "point_count"}, axis=1)
            .reset_index()
            .query(f"port == {pair[0]}") # will be the same..only need one copy
            .drop("port", axis=1)
        )
    
        # configure random sampling
        board_count = pair_boards.shape[0]
        sample_size = min(board_count, boards_sampled)

        # bias toward selecting boards with more overlap
        sample_weight = pair_boards["point_count"] ** 2 

        # get the randomly selected subset
        selected_boards = pair_boards.sample(
            n=sample_size, weights=sample_weight, random_state=random_state
        )
        
        selected_pair_points = pair_points.merge(selected_boards,"right","sync_index")
        
        return selected_pair_points


    def stereo_calibrate(self, pair):
        """Iterates across all camera pairs. Intrinsic parameters are pulled
        from camera and combined with obj and img points for each pair.
        """
        logger.info(f"About to stereocalibrate pair {pair}")

        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

        camA = self.synchronizer.streams[pair[0]].camera
        camB = self.synchronizer.streams[pair[1]].camera

        # pull calibration parameters for pair
        obj = self.stereo_inputs[pair]["common_board_loc"]
        img_A = self.stereo_inputs[pair]["img_loc_A"]
        img_B = self.stereo_inputs[pair]["img_loc_B"]

        # convert to list of vectors for OpenCV function
        obj = [np.array(x, dtype=np.float32) for x in obj]
        img_A = [np.array(x, dtype=np.float32) for x in img_A]
        img_B = [np.array(x, dtype=np.float32) for x in img_B]

        (
            ret,
            camera_matrix_1,
            distortion_1,
            camera_matrix_2,
            distortion_2,
            rotation,
            translation,
            essential,
            fundamental,
        ) = cv2.stereoCalibrate(
            obj,
            img_A,
            img_B,
            camA.camera_matrix,
            camA.distortion,
            camB.camera_matrix,
            camB.distortion,
            imageSize=None,  # this does not matter. from OpenCV: "Size of the image used only to initialize the camera intrinsic matrices."
            criteria=criteria,
            flags=stereocalibration_flags,
        )

        self.stereo_outputs[pair] = {
            "grid_count": len(obj),
            "rotation": rotation,
            "translation": translation,
            "RMSE": ret,
        }

        if pair in self.pairs:
            logger.info(f"Removing pair {pair}")
            self.pairs.remove(pair)
        else:
            logger.warning(f"Attempted to remove pair {pair} but it was not present")
            
        logger.info(
            f"For camera pair {pair}, rotation is \n{rotation}\n and translation is \n{translation}"
        )
        logger.info(f"RMSE of reprojection is {ret}")

    def get_stereocal_inputs(self,port, stereo_pair_data):
        
        port_point_data = paired_point_data .query(f"port == {port}")
    
        sync_indices = port_point_data["sync_index"].to_numpy().round().astype(int)
        img_loc_x = port_point_data["img_loc_x"].to_numpy().astype(np.float32)
        img_loc_y = port_point_data["img_loc_y"].to_numpy().astype(np.float32)
        board_loc_x = port_point_data["board_loc_x"].to_numpy().astype(np.float32)
        board_loc_y = port_point_data["board_loc_y"].to_numpy().astype(np.float32)
        board_loc_z = board_loc_x * 0

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

        return img_locs,board_locs
 
# if __name__ == "__main__":
if True:
    from pathlib import Path

    # set inputs
    session_path = Path(__root__, "tests", "5_cameras")

    config_path = Path(session_path, "config.toml")
    point_data_path = Path(session_path, "recording", "point_data.csv")

    omnical = OmniCalibrator(config_path, point_data_path,)

    pair = (0,1)
    paired_point_data = omnical.get_stereopair_data(pair, boards_sampled=20)
    img_locs_A, board_locs_A = omnical.get_stereocal_inputs(pair[0],paired_point_data)
    img_locs_B, board_locs_B = omnical.get_stereocal_inputs(pair[1],paired_point_data)

    
# %%
