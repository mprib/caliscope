
#%%

from pyxy3d import __root__
from pathlib import Path

test_dir = Path(__root__, "dev", "sample_sessions", "low_res_laptop")
folders = [item.name for item in test_dir.iterdir() if item.is_dir()]
recording_folders = [folder for folder in folders if folder.startswith("recording_")]
recording_counts = [folder.split("_")[1] for folder in recording_folders]
print(recording_counts)
recording_counts = [int(rec_count) for rec_count in recording_counts if rec_count.isnumeric()]
print(recording_counts)
next_recording_count = max(recording_counts)+1

next_directory = "recording_" +str(next_recording_count)
print(next_directory)
# %%
