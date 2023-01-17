#%%

from pathlib import Path
import pickle
import sys
import numpy as np

repo = str(Path(__file__)).split("src")[0]
sys.path.insert(0, repo) # helpful for interactice jupyter session


session_directory = Path(repo, "sessions", "iterative_adjustment")
print(repo)
optimized_path = Path(session_directory, "recording", "optimized_params.pkl")

with open(optimized_path, "rb") as file:
    optimized = pickle.load(file)

# %%
n_points = optimized.fun.shape[0]




