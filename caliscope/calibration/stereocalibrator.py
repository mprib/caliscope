from pathlib import Path
import caliscope.logger
import cv2
import pandas as pd
from caliscope import __root__
import numpy as np
import rtoml
from itertools import combinations

logger = caliscope.logger.get(__name__)


class StereoCalibrator:
    def __init__(
        self,
        config_path: Path,
        point_data_path: Path,
    ):
        self.config_path = config_path
        self.config = rtoml.load(config_path)

        self.ports = []
        # set ports keeping in mind that something may be flagged for ignore
        for key, value in self.config.items():
            if key[0:4] == "cam_":
                # NOTE: Commenting out this line as "ignore" was only a property needed when managing webcams but does not make sense here
                # if not self.config[key]["ignore"]:
                self.ports.append(int(key[4:]))

        # import point data, adding coverage regions to each port
        raw_point_data = pd.read_csv(point_data_path)
        self.all_point_data = self.points_with_coverage_region(raw_point_data)
        self.all_boards = self.get_boards_with_coverage()

        # self.ports = [int(key[4:]) for key in self.config.keys() if key[0:3] == "cam"]
        self.pairs = [(i, j) for i, j in combinations(self.ports, 2) if i < j]

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

    def get_stereopair_data(
        self, pair: tuple, boards_sampled: int, random_state=1
    ) -> pd.DataFrame or None:
        # convenience function to get the points that are in the overlap regions of the pairs
        def in_pair(row: int, pair: tuple):
            """
            Uses the coverage_region string generated previously to flag points that are in
            a shared region of the pair
            """
            a, b = pair
            port_check = row.port == a or row.port == b

            a, b = str(a), str(b)

            region_check = ("_" + a + "_") in row.coverage_region and (
                "_" + b + "_"
            ) in row.coverage_region
            return region_check and port_check

        # flag the points that belong to the pair overlap regions
        self.all_point_data["in_pair"] = self.all_point_data.apply(
            in_pair, axis=1, args=(pair,)
        )

        # group points into boards and get the total count for sample weighting below
        pair_points = self.all_point_data[self.all_point_data["in_pair"] == True]
        pair_boards = (
            pair_points.filter(["sync_index", "port", "point_id"])
            .groupby(["sync_index", "port"])
            .agg("count")
            .rename({"point_id": "point_count"}, axis=1)
            .query("point_count >=6")  # a requirement of the stereocalibration function
            .reset_index()
            .query(f"port == {pair[0]}")  # will be the same..only need one copy
            .drop("port", axis=1)
        )

        # configure random sampling. If you have too few boards, then only take what you have
        board_count = pair_boards.shape[0]
        sample_size = min(board_count, boards_sampled)

        if sample_size > 0:
            logger.info(f"Assembling {sample_size} shared boards for pair {pair}")
            # bias toward selecting boards with more overlapping points
            sample_weight = pair_boards["point_count"] ** 2

            # get the randomly selected subset
            selected_boards = pair_boards.sample(
                n=sample_size, weights=sample_weight, random_state=random_state
            )

            selected_pair_points = pair_points.merge(
                selected_boards, "right", "sync_index"
            )
        else:
            logger.info(f"For pair {pair} there are no shared boards")
            selected_pair_points = None

        return selected_pair_points

    def stereo_calibrate_all(self, boards_sampled=10):
        """Iterates across all camera pairs. Intrinsic parameters are pulled
        from camera and combined with obj and img points for each pair.
        """
        logger.info("Deleting previous stereocalibrations from config")
        # clear out the previous stereocalibrations
        for key in self.config.copy().keys():
            if key[0:6] == "stereo":
                del self.config[key]

        logger.info(f"Beginning stereocalibration of pairs {self.pairs}")
        for pair in self.pairs:
            error, rotation, translation = self.stereo_calibrate(pair, boards_sampled)

            if error is not None:
                # only store data if there was sufficient stereopair coverage to get
                # a good calibration

                # toml dumps arrays as strings, so needs to be converted to list
                rotation = rotation.tolist()
                translation = translation.tolist()

                config_key = "stereo_" + str(pair[0]) + "_" + str(pair[1])
                self.config[config_key] = {}
                self.config[config_key]["rotation"] = rotation
                self.config[config_key]["translation"] = translation
                self.config[config_key]["RMSE"] = error

        logger.info(
            "Direct stereocalibration complete for all pairs for which data is available"
        )
        logger.info(f"Saving stereo-pair extrinsic data to {self.config_path}")
        with open(self.config_path, "w") as f:
            rtoml.dump(self.config, f)

    def stereo_calibrate(self, pair, boards_sampled=10):
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.000001)

        paired_point_data = self.get_stereopair_data(pair, boards_sampled)

        if paired_point_data is not None:
            img_locs_A, obj_locs_A = self.get_stereocal_inputs(
                pair[0], paired_point_data
            )
            img_locs_B, obj_locs_B = self.get_stereocal_inputs(
                pair[1], paired_point_data
            )

            camera_matrix_A = self.config["cam_" + str(pair[0])]["matrix"]
            camera_matrix_B = self.config["cam_" + str(pair[1])]["matrix"]
            camera_matrix_A = np.array(camera_matrix_A, dtype=float)
            camera_matrix_B = np.array(camera_matrix_B, dtype=float)

            distortion_A = self.config["cam_" + str(pair[0])]["distortions"]
            distortion_B = self.config["cam_" + str(pair[1])]["distortions"]
            distortion_A = np.array(distortion_A, dtype=float)
            distortion_B = np.array(distortion_B, dtype=float)

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
                obj_locs_A,
                img_locs_A,
                img_locs_B,
                camera_matrix_A,
                distortion_A,
                camera_matrix_B,
                distortion_B,
                imageSize=None,  # this does not matter. from OpenCV: "Size of the image used only to initialize the camera intrinsic matrices."
                criteria=criteria,
                flags=stereocalibration_flags,
            )

            logger.info(f"RMSE of reprojection for pair {pair} is {ret}")

        else:
            logger.info(f"No stereocalibration produced for pair {pair}")
            ret = None
            rotation = None
            translation = None

        return ret, rotation, translation

    def get_stereocal_inputs(self, port, point_data):
        port_point_data = point_data.query(f"port == {port}")

        sync_indices = port_point_data["sync_index"].to_numpy().round().astype(int)
        img_loc_x = port_point_data["img_loc_x"].to_numpy().astype(np.float32)
        img_loc_y = port_point_data["img_loc_y"].to_numpy().astype(np.float32)
        obj_loc_x = port_point_data["obj_loc_x"].to_numpy().astype(np.float32)
        obj_loc_y = port_point_data["obj_loc_y"].to_numpy().astype(np.float32)
        obj_loc_z = obj_loc_x * 0

        # build the actual inputs for the calibration...
        img_x_y = np.vstack([img_loc_x, img_loc_y]).T
        board_x_y_z = np.vstack([obj_loc_x, obj_loc_y, obj_loc_z]).T

        # print(time.time())
        img_locs = []
        obj_locs = []
        for sync_index in np.unique(sync_indices):
            same_frame = sync_indices == sync_index
            img_locs.append(img_x_y[same_frame])
            obj_locs.append(board_x_y_z[same_frame])

        return img_locs, obj_locs


if __name__ == "__main__":
    # if True:
    from pathlib import Path

    # set inputs
    # session_path = Path(__root__, "tests", "4_cameras_nonoverlap")
    session_path = Path(__root__, "dev", "sample_sessions", "257")

    config_path = Path(session_path, "config.toml")
    point_data_path = Path(session_path, "calibration", "extrinsic", "xy.csv")

    stereocal = StereoCalibrator(
        config_path,
        point_data_path,
    )

    # %%

    stereocal.stereo_calibrate_all(boards_sampled=15)
