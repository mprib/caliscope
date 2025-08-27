from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rtoml

import caliscope.logger

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
                if "ignore" not in value.keys():
                    self.ports.append(int(key[4:]))
                else:
                    if not value["ignore"]:
                        self.ports.append(int(key[4:]))

        # import point data, adding coverage regions to each port
        raw_point_data = pd.read_csv(point_data_path)
        self.all_point_data = self._points_with_coverage_region(raw_point_data)
        self.all_boards = self._get_boards_with_coverage()

        # self.ports = [int(key[4:]) for key in self.config.keys() if key[0:3] == "cam"]
        self.pairs = [(i, j) for i, j in combinations(self.ports, 2) if i < j]

    def _points_with_coverage_region(self, point_data: pd.DataFrame):
        """
        Efficiently create coverage region strings for points.
        """
        # Extract unique combinations of sync_index, point_id, and port
        point_ports = point_data[["sync_index", "point_id", "port"]].drop_duplicates()

        # Convert port to strings for easier handling
        point_ports["port_str"] = point_ports["port"].astype(str)

        # Group by sync_index and point_id to collect ports
        grouped = (
            point_ports.groupby(["sync_index", "point_id"])["port_str"]
            .apply(lambda x: "_" + "_".join(sorted(x)) + "_")
            .reset_index(name="coverage_region")
        )

        # Merge back with original data
        result = point_data.merge(grouped, on=["sync_index", "point_id"], how="left")

        return result

    def _get_boards_with_coverage(self):
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
            self.all_point_data.filter(["port", "sync_index", "point_id", "coverage_region"])
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
            .merge(board_points, "left", ["port", "sync_index"])  # merge in a point_count that means something
        )

        return all_boards

    def get_stereopair_data(self, pair: tuple, boards_sampled: int):
        """
        Efficiently extract data for a stereo pair with deterministic board selection.
        """
        a, b = pair
        a_str, b_str = str(a), str(b)

        # Create vectorized masks for filtering
        in_region_a = self.all_point_data["coverage_region"].str.contains(f"_{a_str}_")
        in_region_b = self.all_point_data["coverage_region"].str.contains(f"_{b_str}_")
        in_port_pair = self.all_point_data["port"].isin([a, b])

        # Filter points that are in the shared region and in one of the cameras
        pair_points = self.all_point_data[in_region_a & in_region_b & in_port_pair].copy()

        if pair_points.empty:
            logger.info(f"For pair {pair} there are no shared points")
            return None

        # Count points per board and filter to boards with enough points
        board_counts = pair_points.groupby(["sync_index", "port"]).size().reset_index(name="point_count")
        valid_boards = board_counts[board_counts["point_count"] >= 6]

        # Filter to port a only (to avoid duplicates)
        valid_boards_a = valid_boards[valid_boards["port"] == a][["sync_index", "point_count"]]

        if valid_boards_a.empty:
            logger.info(f"For pair {pair} there are no boards with sufficient points")
            return None

        # Sample boards deterministically
        sample_size = min(len(valid_boards_a), boards_sampled)

        if sample_size > 0:
            logger.info(f"Assembling {sample_size} shared boards for pair {pair}")

            # Deterministic selection with temporal and quality diversity
            selected_boards = self._select_diverse_boards(valid_boards_a, sample_size)

            # Filter points to selected boards
            selected_points = pair_points[pair_points["sync_index"].isin(selected_boards["sync_index"])]
            return selected_points
        else:
            logger.info(f"For pair {pair} there are no shared boards")
            return None

    def _select_diverse_boards(self, valid_boards_a: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """
        Deterministically select boards with temporal and quality diversity.

        Strategy:
        1. Sort boards by quality (point_count) descending
        2. Apply temporal diversity by selecting boards spread across time
        3. Ensure deterministic ordering by sorting by sync_index as tiebreaker
        """
        # Ensure deterministic ordering
        boards_sorted = valid_boards_a.sort_values(["point_count", "sync_index"], ascending=[False, True]).reset_index(
            drop=True
        )

        if len(boards_sorted) <= sample_size:
            return boards_sorted

        # For temporal diversity, try to spread selection across the time range
        # This gives us boards from different temporal periods
        sync_indices = boards_sorted["sync_index"].values
        min_sync, max_sync = sync_indices.min(), sync_indices.max()

        # Create temporal bins and select best board from each bin
        if sample_size > 1 and max_sync > min_sync:
            # Create time-based selection with quality preference
            selected_indices = []

            # Divide the temporal range into bins
            time_bins = np.linspace(min_sync, max_sync + 1, sample_size + 1)

            for i in range(sample_size):
                # Find boards in this temporal bin
                bin_start, bin_end = time_bins[i], time_bins[i + 1]
                bin_mask = (boards_sorted["sync_index"] >= bin_start) & (boards_sorted["sync_index"] < bin_end)
                bin_boards = boards_sorted[bin_mask]

                if len(bin_boards) > 0:
                    # Select the best quality board from this bin (already sorted by quality)
                    selected_indices.append(bin_boards.index[0])

            # If we didn't get enough boards from temporal binning, fill with remaining best quality boards
            remaining_needed = sample_size - len(selected_indices)
            if remaining_needed > 0:
                available_indices = [idx for idx in boards_sorted.index if idx not in selected_indices]
                selected_indices.extend(available_indices[:remaining_needed])

            # Ensure we have exactly sample_size boards
            selected_indices = selected_indices[:sample_size]
            return boards_sorted.loc[selected_indices]

        else:
            # Simple case: just take the best quality boards
            return boards_sorted.head(sample_size)

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

        logger.info("Direct stereocalibration complete for all pairs for which data is available")
        logger.info(f"Saving stereo-pair extrinsic data to {self.config_path}")
        with open(self.config_path, "w") as f:
            rtoml.dump(self.config, f)

    def stereo_calibrate(self, pair, boards_sampled=10):
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

        paired_point_data = self.get_stereopair_data(pair, boards_sampled)

        if paired_point_data is not None:
            img_locs_A, obj_locs_A = self.get_stereocal_inputs(pair[0], paired_point_data)
            img_locs_B, obj_locs_B = self.get_stereocal_inputs(pair[1], paired_point_data)

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
                imageSize=None,
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
