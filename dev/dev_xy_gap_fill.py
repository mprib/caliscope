


import pandas as pd
import numpy as np

# Set the maximum size of a gap that will be filled
GAP_SIZE_TO_FILL = 3  # Change this value as needed

# Load the data
data = pd.read_csv('/path/to/xy_HOLISTIC_OPENSIM.csv')

# Initialize a DataFrame to store the data with filled gaps
data_filled = pd.DataFrame()

# Loop through each combination of port and point_id
for (port, point_id), group in data.groupby(['port', 'point_id']):
    # Sort by frame_index to ensure the data is in order
    group = group.sort_values('frame_index')
    
    # Calculate the differences between consecutive frame_indices
    group['frame_gap'] = group['frame_index'].diff()
    
    # Identify the gaps that are less than or equal to the specified size
    gaps_to_fill = group['frame_gap'].between(2, GAP_SIZE_TO_FILL + 1)
    
    # Interpolate the values for img_loc_x and img_loc_y
    group.loc[gaps_to_fill, 'img_loc_x'] = group['img_loc_x'].interpolate(method='linear').round().astype(int)
    group.loc[gaps_to_fill, 'img_loc_y'] = group['img_loc_y'].interpolate(method='linear').round().astype(int)
    
    # Append to the overall DataFrame
    data_filled = data_filled.append(group)

# Remove the 'frame_gap' column
data_filled.drop('frame_gap', axis=1, inplace=True)

# Write the DataFrame to a new CSV file
data_filled.to_csv('/path/to/xy_HOLISTIC_OPENSIM_gaps_filled.csv', index=False)
