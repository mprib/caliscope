#%%

from pathlib import Path
import pandas as pd
from calicam import __root__
import seaborn as sns

session_directory = Path(__root__, "tests", "demo")


data_2d = pd.read_csv(Path(session_directory,"data_2d.csv"))
distance_error = pd.read_csv(Path(session_directory,"distance_error.csv"))


#%%
sns.histplot(data = data_2d, x = 'reproj_error')
# %%
facet_reproj_error = sns.FacetGrid(data_2d, col="camera", col_wrap=2)
facet_reproj_error.map_dataframe(sns.histplot, x = "reproj_error")
# %%
facet_distance_error = sns.FacetGrid(distance_error,row="board_distance")
facet_distance_error.map_dataframe(sns.histplot, x = "Distance_Error_mm")



# %%

by_board_distance = (distance_error
                     .filter(["Distance_Error_mm", "Distance_Error_mm_abs", "world_distance", "board_distance"])
                     .groupby("board_distance")
                     .mean()
            )

# %%
