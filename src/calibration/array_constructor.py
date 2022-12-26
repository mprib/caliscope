import pandas as pd
import toml
import sys
from pathlib import Path
import numpy as np


class ArrayConstructor:
    def __init__(self, calibration_data):
        self.initial_array, self.pairs, self.anchor = get_calibration_data(
            calibration_data
        )

        # self.pairs =


def get_calibration_data(calibration_data):
    config = toml.load(Path(calibration_data, "config.toml"))

    daisy_chain = {
        "Pair": [],
        "Primary": [],
        "Secondary": [],
        "error": [],
        "Rotation": [],
        "Translation": [],
    }

    for key, params in config.items():
        if key.split("_")[0] == "stereo":
            port_A = key.split("_")[1]
            port_B = key.split("_")[2]

            pair = (port_A, port_B)
            rotation = np.array(params["rotation"], dtype=np.float64)
            translation = np.array(params["translation"], dtype=np.float64)
            error = float(params["RMSE"])

            daisy_chain["Pair"].append(pair)
            daisy_chain["Primary"].append(port_A)
            daisy_chain["Secondary"].append(port_B)

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

    anchor_camera = mean_error.index[0]
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
    initial_array = initial_array[["Primary", "Secondary", "Rotation", "Translation"]]

    return initial_array, all_pairs, anchor_camera


#%%
if __name__ == "__main__":
    repo = str(Path(__file__)).split("src")[0]

    sys.path.insert(0, repo)
    calibration_data = Path(repo, "sessions", "iterative_adjustment")
    array = ArrayConstructor(calibration_data)
    # init_array = get_initial_array(calibration_data)
    print(array.initial_array)
    print(array.pairs)
# %%
