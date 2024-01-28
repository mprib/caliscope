
#%%
from plotnine import *
# from caliscope.logger import get
# logger = get(__name__)
import cv2
from threading import Thread
import time
import polars as pl
import numpy as np
from pathlib import Path


recording_directory = Path(
    r"C:\Users\Mac Prible\OneDrive\caliscope\test_record\recording_1"
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
    
    mean_x = df['img_loc_x'].mean()
    std_x = df['img_loc_x'].std()

    df = (df.with_columns(((c('img_loc_x') - mean_x) / std_x).alias("norm_x"))
            .with_columns(((c('img_loc_y') - mean_y) / std_y).alias("norm_y"))
    )
    return df

def assign_tracked_group_size(df:pl.DataFrame):
    counts = df.groupby("track_id").agg(c("port").count().alias("track_group_size"))
    return df.join(counts, on="track_id") 

# Applying the function in the pipeline
df_sync = (data
            .sort(["port", "point_id", "frame_index"])
            .with_columns((c("frame_index") - c("frame_index").shift()).alias("frame_delta"))
            .groupby(["port", "point_id"]).apply(assign_tracked_group)
            .with_columns(pl.concat_str([c("port"), c("point_id"), c("tracked_subgroup")], separator="_").alias("track_id"))
            .groupby("track_id").apply(normalize_y_displacement)
            .pipe(assign_tracked_group_size)
           )

df_sync

#%%

df_plt =(df_sync
        #  .filter(c("port").is_in([0]))
         .filter(c("point_id")==4)
         .filter(c("track_group_size")>40)
        #  .to_pandas()

        )
df_plt.write_csv(Path(recording_directory,"inspect.csv" ))
plot = (ggplot(df_plt)+
        aes(x="frame_time", y = "norm_y", color = "track_id")+
        facet_grid("port ~ point_id")+
        geom_point()
        # theme(legend_position='none')
)
plot
# %%
