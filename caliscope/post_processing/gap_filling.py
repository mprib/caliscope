import pandas as pd
import numpy as np

import caliscope.logger
logger = caliscope.logger.get(__name__)


def gap_fill_xy(xy_base:pd.DataFrame, max_gap_size=3) -> pd.DataFrame:
    """
    xy_base: dataframe which should contain the following columns:
        sync_index (or frame_index)
        frame_time	
        port	
        point_id	
        img_loc_x	
        img_loc_y	
    
    Note: can contain other fields that will remain in the data but will not be processed.
    Examples: 
        obj_loc_x	
        obj_loc_y
    
    Note that this can take one or multiple ports, though the port field must be included even if only
    1 port is being examined.
    """
    # Initialize a DataFrame to store the data with filled gaps
    xy_filled = pd.DataFrame()

    # if "frame_index" in xy_base.columns:
    #     index_key = "frame_index"
    # else:
    #     index_key = "sync_index"
    # index_key = "frame_index"
    index_key = "sync_index"

    last_port = -1 # init to impossible value to kick off log event on first run

    # Loop through each combination of port and point_id
    for (port, point_id), group in xy_base.groupby(['port', 'point_id']):
        #### some overhead to allow sparse logging only on the first run through
        if last_port != port:
            logger.info(f"Gap filling for (x,y) data from port {port}. Filling gaps that are {max_gap_size} frames or less...")
        last_port = port
        ### End Conditional Logging
        
        # Sort by frame_index to ensure the data is in order
        group = group.sort_values(index_key)
    
        # Create a new DataFrame with all frame_index values in the range
        all_frames = pd.DataFrame({index_key: np.arange(group[index_key].min(), group[index_key].max() + 1)})
        all_frames['port'] = port
        all_frames['point_id'] = point_id

        # Merge the actual data with the all_frames DataFrame
        merged = pd.merge(all_frames, group, on=['port', 'point_id', index_key], how='left')

        # Calculate the size of each gap
        merged['gap_size'] = merged['img_loc_x'].isnull().astype(int).groupby((merged['img_loc_x'].notnull()).cumsum()).cumsum()

        # Remove the rows where the gap is larger than GAP_SIZE_TO_FILL
        merged = merged[merged['gap_size'] <= max_gap_size]

        # Interpolate the values for img_loc_x and img_loc_y, limit the interpolation to GAP_SIZE_TO_FILL
        merged['frame_time'] = merged['frame_time'].interpolate(method='linear', limit=max_gap_size).astype('float64')
        merged['img_loc_x'] = merged['img_loc_x'].interpolate(method='linear', limit=max_gap_size).astype('float64')
        merged['img_loc_y'] = merged['img_loc_y'].interpolate(method='linear', limit=max_gap_size).astype('float64')

        # Append to the overall DataFrame
        xy_filled = pd.concat([xy_filled, merged])

    logger.info("(x,y) gap filling complete")
    return xy_filled

    

def gap_fill_xyz(xyz_base:pd.DataFrame, max_gap_size=3) -> pd.DataFrame:
    """
    xyz_base: dataframe which should contain the following columns:
        sync_index 
        point_id	
        loc_x	
        loc_y	
        loc_z
    """
    # Initialize a DataFrame to store the data with filled gaps
    xyz_filled = pd.DataFrame()

    # Loop through each combination of port and point_id
    for point_id, group in xyz_base.groupby('point_id'):
        # Sort by frame_index to ensure the data is in order
        group = group.sort_values("sync_index")
    
        # Create a new DataFrame with all frame_index values in the range
        all_frames = pd.DataFrame({"sync_index": np.arange(group["sync_index"].min(), group["sync_index"].max() + 1)})
        all_frames['point_id'] = point_id

        # Merge the actual data with the all_frames DataFrame
        merged = pd.merge(all_frames, group, on=['point_id', "sync_index"], how='left')

        # Calculate the size of each gap
        merged['gap_size'] = merged['x_coord'].isnull().astype(int).groupby((merged['x_coord'].notnull()).cumsum()).cumsum()

        # Remove the rows where the gap is larger than GAP_SIZE_TO_FILL
        merged = merged[merged['gap_size'] <= max_gap_size]

        # Interpolate the values for img_loc_x and img_loc_y, limit the interpolation to GAP_SIZE_TO_FILL
        merged['x_coord'] = merged['x_coord'].interpolate(method='linear', limit=max_gap_size).astype('float64')
        merged['y_coord'] = merged['y_coord'].interpolate(method='linear', limit=max_gap_size).astype('float64')
        merged['z_coord'] = merged['z_coord'].interpolate(method='linear', limit=max_gap_size).astype('float64')

        # Append to the overall DataFrame
        xyz_filled = pd.concat([xyz_filled, merged])

    return xyz_filled