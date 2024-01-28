#%%

from pathlib import Path
import pandas as pd
from caliscope import __root__
import seaborn as sns
import numpy as np

session_directory = Path(__root__, "tests", "demo")


data_2d = pd.read_csv(Path(session_directory,"data_2d.csv"))
distance_error = pd.read_csv(Path(session_directory,"distance_error.csv"))


# %%
facet_reproj_error = sns.FacetGrid(data_2d, col="camera", col_wrap=2)
facet_reproj_error.map_dataframe(sns.boxplot, x = "stage", y = "reproj_error", showfliers=False)
# %%
facet_distance_error = sns.FacetGrid(distance_error,row="board_distance")
facet_distance_error.map_dataframe(sns.boxplot, x="stage", y = "Distance_Error_mm", showfliers=False)
# sns.boxplot(data = distance_error, x = "stage", y = "Distance_Error_mm_abs", showfliers=False)
#%%
facet_distance_error = sns.FacetGrid(distance_error,row="board_distance")
facet_distance_error.map_dataframe(sns.boxplot, x="stage", y = "percent_match", showfliers=False)

# %%

by_board_distance = (distance_error
                     .filter(["Distance_Error_mm", "Distance_Error_mm_abs", "world_distance", "board_distance"])
                     .groupby("board_distance")
                     .mean()
            )

# %%

rmse = (data_2d.filter(["reproj_error_sq"])
                  .mean())

rmse_by_camera = (data_2d.filter(["camera", "reproj_error_sq"])
                  .groupby("camera")
                  .mean(["reproj_error_sq"])
                  .rename(columns={"reproj_error_sq":"mean_sq_error"}))
rmse_by_camera["rmse"]=np.sqrt(rmse_by_camera["mean_sq_error"])

#%%

sns.boxplot(data = distance_error, 
            x="board_distance", 
            y= "Distance_Error_mm_abs",
            showfliers = False

            )
# %%
