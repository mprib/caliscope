import cv2 as cv
from charuco import Charuco
import numpy as np


# from https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
# The next step is to obtain the projection matrices. This is done simply by multiplying the camera matrix by the rotation and translation matrix.
# NOTE: This is going to be a real messy prototype while I figure out how in the 
# hell to make this work


# read in the 



#RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtx2 @ RT2 #projection matrix for C2

from scipy import linalg

def DLT(P1, P2, point1, point2):
    """
    From https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    """
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


uvs1 = [[458, 86], [451, 164], [287, 181],
        [196, 383], [297, 444], [564, 194],
        [562, 375], [596, 520], [329, 620],
        [488, 622], [432, 52], [489, 56]]
 
uvs2 = [[540, 311], [603, 359], [542, 378],
        [525, 507], [485, 542], [691, 352],
        [752, 488], [711, 605], [549, 651],
        [651, 663], [526, 293], [542, 290]]
 
uvs1 = np.array(uvs1)
uvs2 = np.array(uvs2)
 

p3ds = []
for uv1, uv2 in zip(uvs1, uvs2):
    _p3d = DLT(P1, P2, uv1, uv2)
    p3ds.append(_p3d)
p3ds = np.array(p3ds)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-15, 5)
ax.set_ylim3d(-10, 10)
ax.set_zlim3d(10, 30)
 
connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
for _c in connections:
    print(p3ds[_c[0]])
    print(p3ds[_c[1]])
    ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
 
plt.show()