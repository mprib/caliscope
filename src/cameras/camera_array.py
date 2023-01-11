import pandas as pd
import toml
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import cv2

@dataclass
class CameraData:
    """A place to hold the calibration data associated with a camera that has been populated from a config file.
    For use with final setting of the array and triangulation, but no actual camera management.
    """

    port: int
    resolution: tuple
    camera_matrix: np.ndarray
    error: float
    distortion: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    
    def to_vector(self):
        """
        Converts camera parameters to a numpy vector for use with bundle adjustment.
        This will undergo refactoring along with the bundle adjustment
        """

        # rotation of the camera relative to the world
        rotation_matrix_world = self.rotation

        # rotation of the world relative to camera
        rotation_matrix_proj = np.linalg.inv(rotation_matrix_world)

        rotation_rodrigues = cv2.Rodrigues(rotation_matrix_proj)[0]  # elements 0,1,2

        translation_world = self.translation  # elements 3,4,5
        translation_proj = translation_world * -1
        # two focal lengths for potentially rectangular pixels...
        # I'm assuming they are square
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        f = (fx + fy) / 2  # element 6

        # get k1 and k2 from distortion
        k1 = self.distortion[0, 0]  # element 7
        k2 = self.distortion[0, 1]  # element 8

        port_param = np.hstack(
            [rotation_rodrigues[:, 0], translation_proj[:, 0], f, k1, k2]
        )
        
        return port_param
        
    def from_vector(self, row):
        """
        Takes a vector in the same format that is output and updates the camera 
        with those parameters
        """
        self.rotation = cv2.Rodrigues(row[0:3])[0]  # note it is first element
        self.translation = row[3:6]
        f = row[6]
        k1 = row[7]
        k2 = row[8]
        self.camera_matrix[0,0] = f
        self.camera_matrix[1,1] = f
        self.distortion[0,0] = k1
        self.distortion[0,1] = k2

@dataclass
class CameraArray:
    """The plan is that this will expand to become and interface for setting the origin.
    At the moment all it is doing is holding a dictionary of CameraData objects"""
    cameras: dict
    
class CameraArrayBuilder:
    """An ugly class to wrangle the config data into a useful set of camera
    data objects in a common frame of reference and then return it as a CameraArray object;
    Currently set up to only work when all possible stereo pairs have been stereocalibrated;
    I anticipate this is something that will need to be expanded in the future to account
    for second order position estimates. This might be solved cleanly by putting a frame of reference
    in the CameraData object and extending the CameraArray to place elements in a common frame of reference"""

    def __init__(self, config_path: Path):

        self.config = toml.load(config_path)
        self.extrinsics, self.pairs, self.anchor = self.get_extrinsic_data()
        self.set_cameras()

    def set_cameras(self):
        self.cameras = {}
        # create the basic cameras based off of intrinsic params stored in config
        for key, data in self.config.items():
            if key.startswith("cam_"):
                port = data["port"]
                resolution = data["resolution"]
                camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
                error = data["error"]
                distortion = np.array(data["distortion"], dtype=np.float64)

                # update with extrinsics, though place anchor camera at origin
                if port == self.anchor:
                    translation = np.array([[0], [0], [0]], dtype=np.float64)
                    rotation = np.array(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
                    )
                else:
                    anchored_pair = self.extrinsics.query(f"Secondary == {port}")
                    translation = anchored_pair.Translation.to_list()[0]
                    rotation = anchored_pair.Rotation.to_list()[0]

                cam_data = CameraData(
                    port,
                    resolution,
                    camera_matrix,
                    error,
                    distortion,
                    translation,
                    rotation,
                )
                self.cameras[port] = cam_data

    def get_extrinsic_data(self):

        daisy_chain = {
            "Pair": [],
            "Primary": [],
            "Secondary": [],
            "error": [],
            "Rotation": [],
            "Translation": [],
        }

        for key, params in self.config.items():
            if key.split("_")[0] == "stereo":
                port_A = int(key.split("_")[1])
                port_B = int(key.split("_")[2])

                pair = (port_A, port_B)
                rotation = np.array(params["rotation"], dtype=np.float64)
                translation = np.array(params["translation"], dtype=np.float64)
                error = float(params["RMSE"])

                daisy_chain["Pair"].append(pair)
                
                # it will likely appear strange to make B the primary and A the secondary
                # because cv2.stereocalibrate returns R and t such that it is the 
                # position of the first camera relative to the second camera, I have 
                # switched things up for purposes of constructing the array
                daisy_chain["Primary"].append(port_B)
                daisy_chain["Secondary"].append(port_A)

                daisy_chain["Rotation"].append(rotation)
                daisy_chain["Translation"].append(translation)
                daisy_chain["error"].append(error)

        daisy_chain = pd.DataFrame(daisy_chain).sort_values("error")

        # create an inverted version of these to determine best Anchor camera
        inverted_chain = daisy_chain.copy()
        inverted_chain.Primary, inverted_chain.Secondary = (
            inverted_chain.Secondary,
            inverted_chain.Primary,
        )
        inverted_chain.Translation = inverted_chain.Translation * -1
        inverted_chain.Rotation = inverted_chain.Rotation.apply(np.linalg.inv)

        daisy_chain_w_inverted = pd.concat([daisy_chain, inverted_chain], axis=0)

        all_pairs = daisy_chain["Pair"].unique()

        mean_error = (
            daisy_chain_w_inverted.filter(["Primary", "error"])
            .groupby("Primary")
            .agg("mean")
            .rename(columns={"error": "MeanError"})
            .sort_values("MeanError")
        )

        anchor_camera = int(mean_error.index[0]) # array anchored by camera with the lowest mean RMSE

        daisy_chain_w_inverted = daisy_chain_w_inverted.merge(
            mean_error, how="left", on="Primary"
        ).sort_values("MeanError")

        daisy_chain_w_inverted.insert(
            4, "MeanError", daisy_chain_w_inverted.pop("MeanError")
        )
        daisy_chain_w_inverted.sort_values(["MeanError"])

        # need to build an array of cameras in a common frame of reference a starting point for the calibration
        # if one of the stereo pairs did not get calibrated, then some additional tricks will need to get
        # deployed to make things work. But fortunately this is the simpler case now.
        initial_array = daisy_chain_w_inverted[
            daisy_chain_w_inverted.Primary == anchor_camera
        ]
        initial_array = initial_array[
            ["Primary", "Secondary", "Rotation", "Translation"]
        ]

        # fix format of Primary/Secondary labels to be integers
        initial_array[["Primary", "Secondary"]] = initial_array[
            ["Primary", "Secondary"]
        ].apply(pd.to_numeric)

        return initial_array, all_pairs, anchor_camera

    def get_camera_array(self):
        return CameraArray(self.cameras)


if __name__ == "__main__":
    repo = str(Path(__file__)).split("src")[0]

    config_path = Path(repo, "sessions", "iterative_adjustment", "config.toml")
    array_builder = CameraArrayBuilder(config_path)
    camera_array = array_builder.get_camera_array()
    
    print("pause")