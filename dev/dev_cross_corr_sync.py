
#%%
from plotnine import *
from pyxy3d.logger import get
import cv2
from threading import Thread
import time
logger = get(__name__)
import polars as pl
import numpy as np
from pathlib import Path


recording_directory = Path(
    r"C:\Users\Mac Prible\OneDrive\pyxy3d\test_record\recording_1"
)

data = pl.read_csv(Path(recording_directory,"combined_gap_filled_alt.csv"))

#%%
c = pl.col

def assign_tracked_group(df:pl.DataFrame):
    change_mask  = df.with_columns((c("frame_delta")!=c("frame_delta").shift()).alias("change_mask"))
    return change_mask.with_columns(c("change_mask").cumsum().alias("tracked_subgroup"))

def normalize_y_displacement(df: pl.DataFrame) -> pl.DataFrame:
    mean_y = df['img_loc_y'].mean()
    std_y = df['img_loc_y'].std()
    df = df.with_columns(((c('img_loc_y') - mean_y) / std_y).alias("norm_y"))
    return df

# Applying the function in the pipeline
df_sync = (data
            .sort(["port", "point_id", "frame_index"])
            .with_columns((c("frame_index") - c("frame_index").shift()).alias("frame_delta"))
            .groupby(["port", "point_id"]).apply(assign_tracked_group)
            .with_columns(pl.concat_str([c("port"), c("point_id"), c("tracked_subgroup")], separator="_").alias("track_id"))
            .groupby("track_id").apply(normalize_y_displacement)
           )

df_sync

#%%

df_plt =(df_sync
        #  .filter(c("port").is_in([0,1]))
         .filter(c("point_id")==7)
        #  .to_pandas()
        )

plot = (ggplot(df_plt)+
        aes(x="frame_time", y = "norm_y", color = "track_id")+
        facet_grid("port ~.")+
        geom_point()
)
plot
# %%
