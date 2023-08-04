
#%%
import matplotlib.pyplot as plt
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


# consolidate all data created by dev_prep_sync_test_data
csv_data = []
for csv_path in recording_directory.glob("*_alt.csv"):
    logger.info(f"Aggregating data from {csv_path}")
    csv_data.append(pl.read_csv(csv_path))

data = pl.read_csv(Path(recording_directory,"combined_gap_filled_alt.csv"))
# for csv_path in recording_directory.glob("*_alt.csv"):
#     if data is None:
#         data = pd.read_csv(csv_path)
        
#     else:
#         current_data = pd.read_csv(csv_path)
#         data = pd.concat([data,current_data])

# for port in data["port"].unique():
#     for point_id in data.query(f"port == {port}")["point_id"].unique():
#         logger.info(f"Normalizing data for port {port} and point_id {point_id}")

                
