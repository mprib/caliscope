#%%

import numpy as np
import cv2

# extrinsic parameters of the two cameras
cam1_rvec = np.array([0.1, 0.2, 0.3])
cam1_tvec = np.array([1.0, 2.0, 3.0])
cam2_rvec = np.array([-0.2, 0.1, -0.4])
cam2_tvec = np.array([0.5, -1.0, 2.0])

# extrinsic parameters of the charuco board relative to the first camera
board_rvec = np.array([0.0, 0.0, 0.0])
board_tvec = np.array([0.1, 0.1, 0.0])

# transformation from charuco board to first camera
print(cv2.composeRT(np.zeros(3), np.zeros(3), board_rvec, board_tvec))

board_RT = cv2.composeRT(np.zeros(3), np.zeros(3), board_rvec, board_tvec)
board_R, board_T = board_RT[0], board_RT[1]

# %%
cam1_R, cam1_T = cv2.composeRT(np.zeros(3), np.zeros(3), cam1_rvec, cam1_tvec)
board_to_cam1_R = np.dot(board_R, cam1_R.T)
board_to_cam1_T = cam1_T - np.dot(board_to_cam1_R, board_T)

# transformation from second camera to charuco board
_, cam2_R, cam2_T = cv2.composeRT(np.zeros(3), np.zeros(3), cam2_rvec, cam2_tvec)
cam2_to_board_R = np.dot(board_to_cam1_R.T, cam2_R)
cam2_to_board_T = np.dot(board_to_cam1_R.T, cam2_T - board_to_cam1_T)

# new extrinsic parameters of the second camera
_, new_cam2_rvec, new_cam2_tvec = cv2.composeRT(np.zeros(3), np.zeros(3), cam2_to_board_R, cam2_to_board_T)

# now, you can use the new extrinsic parameters of both cameras to triangulate points


# %%
