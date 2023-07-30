
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from numba import jit
from numba.typed import Dict, List
import numpy as np

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

# Copyright (c) 2019, Pierre Karashchuk
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