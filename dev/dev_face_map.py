#%%
import plotly.graph_objs as go
import numpy as np
from pyxy3d import __root__
from pathlib import Path
import pandas as pd

# Assuming you have your face landmarks as a list of (x, y, z, name) tuples
# Example: landmarks = [(10, 20, 30, 'nose'), (30, 40, 50, 'left_eye'), ...]
landmarks = np.random.rand(10, 3)  # Replace with your data
# names = ['point_%d' % i for i in range(10)]  # Replace with your names

source_file = Path(__root__, "tests", "sessions_copy_delete","4_cam_recording", "recording_1", "HOLISTIC", "xyz_HOLISTIC.csv")
xyz_data = pd.read_csv(source_file)
xyz_data = xyz_data.query("sync_index == 0")
#%%

trace = go.Scatter3d(
    # swapping axes to shift to upright
    z=xyz_data["x_coord"],
    x=xyz_data["y_coord"],
    y=xyz_data["z_coord"],
    mode='markers',
    marker=dict(
        size=1,
        # color=range(10),  # set color to an array/list of desired values
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    ),
    text=xyz_data["point_id"],
    hoverinfo='text'
)

data = [trace]
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
fig = go.Figure(data=data, layout=layout)
fig.show()

# %%
