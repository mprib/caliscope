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

xy_repoj_error = optimized.fun.reshape(-1,2)
euclidean_distance_error = np.sqrt(np.sum((xy_repoj_error)**2, axis = 1))
rmse_reproj_error = np.sqrt(np.mean(euclidean_distance_error**2))
# coming in at  1.7 pixels... too much, but then it seems like the solution is just to...
# ....uh...drop the ones with a large error. Feels weird.
# %%
euclidean_distance_error.sort()
n_2d_points = xy_repoj_error.shape[0]
percent_cutoff = .8
subset_rmse =  np.sqrt(np.mean(euclidean_distance_error[0:int(percent_cutoff*n_2d_points)]**2))
print(subset_rmse)


error_rank = np.argsort(euclidean_distance_error)

# MAC: Start here tomorrow...trying to figure out how to filter the inputs...
# OK....this is going to become a fun and challenging little problem
# accepted_ranks


# %%
