# Building out the beginnings of something that will be a large part of this whole thing.
# Going to start with video footage from a calibration and the resulting set of mono/stereo calibration
# config file, and then iteratively refine those using methods similar to anipose

#%%
import pandas as pd
import toml
import sys
from pathlib import Path
import numpy as np

repo = str(Path(__file__)).split("src")[0]
sys.path.insert(0, repo)

from src.triangulate.stereo_triangulator import StereoTriangulator

class ArrayConstructor():

    def __init__(self):
        self.session = session        

        
        

if __name__ == "main":
    from src.session import Session 

    session_path = Path(repo, "sessions", "iterative_adjustment")

    session = Session(session_path)
    config = toml.load(Path(session_path, "config.toml"))

daisy_chain = {
    "Pair": [],
    "Config_Key": [],
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

        # new_key = f"pair_{port_A}_{port_B}"
        pair = (port_A, port_B)
        rotation = np.array(params["rotation"], dtype=np.float64)
        translation = np.array(params["translation"], dtype=np.float64)
        error = float(params["RMSE"])


        daisy_chain["Pair"].append(pair)
        daisy_chain["Config_Key"].append(key)
        daisy_chain["Primary"].append(port_A)
        daisy_chain["Secondary"].append(port_B)

        daisy_chain["Rotation"].append(rotation)
        daisy_chain["Translation"].append(translation)
        daisy_chain["error"].append(error)

daisy_chain = pd.DataFrame(daisy_chain).sort_values("error")

#%%
all_pairs = daisy_chain["Pair"].unique()
print(all_pairs)
#%%
# create the inverted formats for all of the rows

inverted_relationships = {
    "Pair": [],
    "Config_Key": [],
    "Primary": [],
    "Secondary": [],
    "error": [],
    "Rotation": [],
    "Translation": [],
}
rows = daisy_chain.shape[0]
for index in range(rows):
    orig_row = daisy_chain.loc[index]
    inverted_relationships["Pair"].append(orig_row["Pair"])
    inverted_relationships["Config_Key"].append(orig_row["Config_Key"])
    inverted_relationships["Primary"].append(orig_row["Secondary"])
    inverted_relationships["Secondary"].append(orig_row["Primary"])
    inverted_relationships["error"].append(orig_row["error"])
    inverted_relationships["Rotation"].append(np.linalg.inv(orig_row["Rotation"]))
    inverted_relationships["Translation"].append(orig_row["Translation"] * -1)

inverted_relationships = pd.DataFrame(inverted_relationships)
daisy_chain = pd.concat([daisy_chain, inverted_relationships])
daisy_chain = daisy_chain.sort_values(["Primary", "Secondary"])

#%%
# get primary with the lowest RMSE connecting it to daisy_chain.

mean_error = (
    daisy_chain.filter(["Primary", "error"])
    .groupby("Primary")
    .agg("mean")
    .rename(columns={"error": "MeanError"})
    .sort_values("MeanError")
)

print(mean_error)

Lowest_Error = mean_error[mean_error.MeanError == mean_error.MeanError.min()]

print(Lowest_Error)
# %%
sorted_chain = daisy_chain.merge(mean_error, how="left", on="Primary").sort_values(
    "MeanError"
)

sorted_chain    
# %%
