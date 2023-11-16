

import pyxy3d.logger
import pandas as pd
from time import time
from numba import jit
from numba.typed import Dict, List
from pyxy3d.cameras.camera_array import CameraArray

import numpy as np
logger = pyxy3d.logger.get(__name__)

# helper function to avoid use of np.unique(return_counts=True) which doesn't work with jit
@jit(nopython=True, cache=True)
def unique_with_counts(arr):
    sorted_arr = np.sort(arr)
    unique_values = [sorted_arr[0]]
    counts = [1]

    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != sorted_arr[i - 1]:
            unique_values.append(sorted_arr[i])
            counts.append(1)
        else:
            counts[-1] += 1

    return np.array(unique_values), np.array(counts)

#####################################################################################
# The following code is adapted from the `Anipose` project, 
# in particular the `triangulate_simple` function of `aniposelib`
# Original author:  Lili Karashchuk
# Project: https://github.com/lambdaloop/aniposelib/
# Original Source Code : https://github.com/lambdaloop/aniposelib/blob/d03b485c4e178d7cff076e9fe1ac36837db49158/aniposelib/cameras.py#L21
# This code is licensed under the BSD 2-Clause License
# BSD 2-Clause License

# Copyright (c) 2019, Lili Karashchuk
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
   # list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
   # this list of conditions and the following disclaimer in the documentation
   # and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@jit(nopython=True, parallel=True, cache=True)
def triangulate_sync_index(
    projection_matrices:Dict, current_camera_indices:np.ndarray, current_point_id:np.ndarray, current_img:np.ndarray
):
    # sync_indices_xyz = List()
    point_indices_xyz = List()
    obj_xyz = List()

    unique_points, point_counts = unique_with_counts(current_point_id)
    for index in range(len(point_counts)):
        if point_counts[index] > 1:
            # triangulate that point...
            point = unique_points[index]
            points_xy = current_img[current_point_id == point]
            camera_ids = current_camera_indices[current_point_id == point]

            num_cams = len(camera_ids)
            A = np.zeros((num_cams * 2, 4))
            for i in range(num_cams):
                x, y = points_xy[i]
                P = projection_matrices[camera_ids[i]]
                A[(i * 2) : (i * 2 + 1)] = x * P[2] - P[0]
                A[(i * 2 + 1) : (i * 2 + 2)] = y * P[2] - P[1]
            u, s, vh = np.linalg.svd(A, full_matrices=True)
            point_xyzw = vh[-1]
            point_xyz = point_xyzw[:3] / point_xyzw[3]

            point_indices_xyz.append(point)
            obj_xyz.append(point_xyz)

    return point_indices_xyz, obj_xyz
# End of adapted code
##################################################################################




def triangulate_xy(xy: pd.DataFrame, camera_array:CameraArray) -> pd.DataFrame:
        
    # assemble numba compatible dictionary
    projection_matrices = camera_array.projection_matrices

    xyz = {
        "sync_index": [],
        "point_id": [],
        "x_coord": [],
        "y_coord": [],
        "z_coord": [],
    }

    sync_index_max = xy["sync_index"].max()

    start = time()
    last_log_update = int(start)  # only report progress each second

    for index in xy["sync_index"].unique():
        active_index = xy["sync_index"] == index

        # load variables for given sync index
        port = xy["port"][active_index].to_numpy()
        point_ids = xy["point_id"][active_index].to_numpy()
        img_loc_x = xy["img_loc_x"][active_index].to_numpy()
        img_loc_y = xy["img_loc_y"][active_index].to_numpy()
        imgs_xy = np.vstack([img_loc_x, img_loc_y]).T

        # the fancy part
        point_id_xyz, points_xyz = triangulate_sync_index(
            projection_matrices, port, point_ids, imgs_xy
        )

        if len(point_id_xyz) > 0:
            # there are points to store so store them...
            xyz["sync_index"].extend([index] * len(point_id_xyz))
            xyz["point_id"].extend(point_id_xyz)

            points_xyz = np.array(points_xyz)
            xyz["x_coord"].extend(points_xyz[:, 0].tolist())
            xyz["y_coord"].extend(points_xyz[:, 1].tolist())
            xyz["z_coord"].extend(points_xyz[:, 2].tolist())

        # only log percent complete each second
        if int(time()) - last_log_update >= 1:
            percent_complete = int(100*(index/sync_index_max))
            logger.info(
                f"(Stage 2 of 2): Triangulation of (x,y) point estimates is {percent_complete}% complete"
            )
            last_log_update = int(time())

    # convert to dataframe prior to returning
    xyz = pd.DataFrame(xyz)
    return xyz
