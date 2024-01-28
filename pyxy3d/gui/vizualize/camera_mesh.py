
import caliscope.logger

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import math

from caliscope.cameras.camera_array import CameraData
import numpy as np
logger = caliscope.logger.get(__name__)

class CameraMesh:
    """Build a camera mesh object that is looking up from the origin"""
    def __init__(self, res, cam_matrix, scale_factor=5000):
        
        # self.scene = scene

        self.width = res[0]/scale_factor
        self.height = res[1]/scale_factor
        self.fx = cam_matrix[0][0]/scale_factor
        self.fy = cam_matrix[1][1]/scale_factor
        self.cx = cam_matrix[0][2]/scale_factor
        self.cy = cam_matrix[1][2]/scale_factor

        self.f_mean = (self.fx+self.fy)/2 # mean focal length...height of inverted pyramid

        self.build_verts()
        self.build_faces()
        self.build_colors()

        self.mesh = gl.GLMeshItem(vertexes=self.verts, 
                                  faces=self.faces, 
                                  faceColors=self.colors, 
                                  smooth=False, 
                                  drawEdges=True, 
                                  edgeColor=(0,0,1,1))
        self.mesh.setGLOptions('additive')

        logger.debug(self.verts)
        logger.debug(self.faces)
        logger.debug(self.colors)

    def build_verts(self):
        right_side_border = self.width-self.cx
        left_side_border = -self.cx
        top_side_border = self.height-self.cy
        bottom_side_border = -self.cy
        pyramid_height = self.f_mean

        self.verts = [[0,    0,     0],   #0: focal point at origin
                      [right_side_border,top_side_border,pyramid_height],         #1: top right of image
                      [right_side_border,bottom_side_border,pyramid_height],         #2: bottom right of image
                      [left_side_border,bottom_side_border,pyramid_height],         #3: bottom left of image
                      [left_side_border,top_side_border,pyramid_height]]         #4: top left of image

        self.verts = np.array(self.verts)      

    def build_faces(self):
        self.faces = [[0,1,2],
                      [0,2,3],
                      [0,3,4],
                      [0,4,1],
                      [1,2,3],
                      [3,4,1]]

        self.faces = np.array(self.faces)

    def build_colors(self):
        self.colors = [[.5,1,0,.2],
                       [.5,1,0,.2],
                       [.5,1,0,.2],
                       [.5,1,0,.2],
                       [0,0,0,.9],
                       [0,0,0,.9]]

        self.colors = np.array(self.colors)        

def rotation_to_float(rotation_matrix):
    new_matrix = []
    for row in rotation_matrix:
        new_row = [float(x) for x in row]
        new_matrix.append(new_row)

    return np.array(new_matrix, dtype=np.float32)


# helper functions to assist with scene creation
def mesh_from_camera(camera_data: CameraData):
    """ "
    Mesh is placed at origin by default. Note that it appears rotations
    are in the mesh frame of reference and translations are in
    the scene frame of reference. I could be wrong, but that appears
    to be the case.

    """
    mesh = CameraMesh(camera_data.size, camera_data.matrix).mesh

    R = camera_data.rotation
    t = camera_data.translation
    camera_orientation_world = R.T

    # rotate mesh
    euler_angles = rotationMatrixToEulerAngles(camera_orientation_world)
    euler_angles_deg = [x * (180 / math.pi) for x in euler_angles]
    x = euler_angles_deg[0]
    y = euler_angles_deg[1]
    z = euler_angles_deg[2]

    # rotate mesh; z,y,x is apparently the order in which it's done
    # https://gamedev.stackexchange.com/questions/16719/what-is-the-correct-order-to-multiply-scale-rotation-and-translation-matrices-f
    mesh.rotate(z, 0, 0, 1, local=True)
    mesh.rotate(y, 0, 1, 0, local=True)
    mesh.rotate(x, 1, 0, 0, local=True)

    camera_origin_world = -np.dot(R.T, t)
    x, y, z = [p for p in camera_origin_world]
    mesh.translate(x, y, z)

    return mesh


def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

if __name__ == '__main__':
    import rtoml
    from pathlib import Path
    
    app = pg.mkQApp("GLMeshItem Example")

    scene = gl.GLViewWidget()
    scene.show()
    scene.setWindowTitle('Camera Calibration')
    scene.setCameraPosition(distance=4)

    # grid = gl.GLGridItem()
    # grid.scale(1,1,1)
    # scene.addItem(grid)
    axis = gl.GLAxisItem()
    scene.addItem(axis)


    repo = Path(str(Path(__file__)).split("pyxy")[0],"pyxy")
    # config_path = r"config 2.toml"
    config = rtoml.load(Path(repo, "sessions", "iterative_adjustment", "config.toml"))
    cams = {}    
    ports = []

    # build cameras
    for key, params in config.items():
        if "cam" in key:

            res = params["resolution"]
            cam_matrix = params["camera_matrix"]
            port = params["port"]
            ports.append(port)
            for row in cam_matrix:
                for index in range(len(row)):
                    row[index] = float(row[index])

            cam_matrix = np.array(cam_matrix, dtype=np.float32)
            cams[port] = CameraMesh(res, cam_matrix)

            print(key)
            print(params)
    
    # need to track frame of reference for each camera posiiton
    # must be able ot iterate over each frame of reference
    # place cameras


    origin_port = 0
    cams[origin_port].mesh.setGLOptions('additive')
    scene.addItem(cams[origin_port].mesh)    

    for key, params in config.items():
        if "stereo" in key:
            pair = key.split("_")[1:3]
            pair = [int(p) for p in pair]
            
            if origin_port in pair:
                for param_key, value in params.items(): 
                    if "translation" in param_key:
                        translation = [float(x[0]) for x in value]
                        translation_scale = 1

                        # reverse translation if origin is the second item
                        if origin_port == pair[1]:
                            translation = [-i for i in translation]
                            other_port = pair[0]
                        else:
                            other_port = pair[1]

                        x,y,z = [t/translation_scale for t in translation]
                        cams[other_port].mesh.translate(x,y,z)
                        logger.info(f"Translation: x: {x}, y: {y}, z: {z}")
                        # cams[other_port].mesh.setGLOptions('additive')

                        scene.addItem(cams[other_port].mesh)
                    if "rotation" in param_key:
                        rotation_count = rotation_to_float(value) # feeding in 3x3 rotation matrix  from config file
                        rotation_count = rotationMatrixToEulerAngles(rotation_count) # convert to angles
                        if origin_port == pair[1]:
                            rotation_count = -rotation_count
                            other_port = pair[0]
                        else:
                            other_port = pair[1]

                        rot_deg = [x*(180/math.pi) for x in rotation_count] # convert to degrees
                        print(f"Rotation (deg) for port {other_port}: {rot_deg}")
                        x = rot_deg[0]
                        y = rot_deg[1]
                        z = rot_deg[2]

                        logger.info(f"ROTATION: x: {x}, y: {y}, z: {z}")
                        cams[other_port].mesh.rotate(x,1,0,0, local=True)
                        cams[other_port].mesh.rotate(y,0,1,0, local=True)
                        cams[other_port].mesh.rotate(z,0,0,1, local=True)
                        # cams[other_port].mesh.setGLOptions('additive')

                        scene.addItem(cams[other_port].mesh)

    pg.exec()
