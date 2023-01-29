import logging

LOG_FILE = "log\camera_array.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

from pathlib import Path
import numpy as np
from dataclasses import dataclass
import cv2
from scipy.optimize import least_squares
import pickle

from calicam.calibration.bundle_adjustment.point_data import PointData

CAMERA_PARAM_COUNT = 6


@dataclass
class CameraData:
    """A place to hold the calibration data associated with a camera that has been populated from a config file.
    For use with final setting of the array and triangulation, but no actual camera management.
    """

    port: int
    resolution: tuple
    camera_matrix: np.ndarray
    error: float
    distortion: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray

    def extrinsics_to_vector(self):
        """
        Converts camera parameters to a numpy vector for use with bundle adjustment.
        """
        # rotation of the camera relative to the world
        rotation_matrix_world = self.rotation

        # rotation of the world relative to camera
        rotation_matrix_proj = np.linalg.inv(rotation_matrix_world)
        rotation_rodrigues = cv2.Rodrigues(rotation_matrix_proj)[0]  # elements 0,1,2
        translation_world = self.translation  # elements 3,4,5
        translation_proj = translation_world * -1

        port_param = np.hstack([rotation_rodrigues[:, 0], translation_proj[:, 0]])

        return port_param

    def extrinsics_from_vector(self, row):
        """
        Takes a vector in the same format that is output of `extrinsics_to_vector` and updates the camera
        """

        # convert back to world frame of reference
        self.rotation = np.linalg.inv(cv2.Rodrigues(row[0:3])[0])
        self.translation = np.array([row[3:6] * -1], dtype=np.float64).T


@dataclass
class CameraArray:
    """The plan is that this will expand to become and interface for setting the origin.
    At the moment all it is doing is holding a dictionary of CameraData objects"""

    cameras: dict

    def get_extrinsic_params(self):
        """for each camera build the CAMERA_PARAM_COUNT element parameter index
        camera_params with shape (n_cameras, CAMERA_PARAM_COUNT)
        contains initial estimates of parameters for all cameras.
        First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
        next 3 components form a translation vector
        """
        camera_params = None
        for port, cam in self.cameras.items():
            port_param = cam.extrinsics_to_vector()
            if camera_params is None:
                camera_params = port_param
            else:
                camera_params = np.vstack([camera_params, port_param])

        return camera_params

    def update_extrinsic_params(self, optimized_x):

        n_cameras = len(self.cameras)
        n_cam_param = 6  # 6 DoF
        flat_camera_params = optimized_x[0 : n_cameras * n_cam_param]
        new_camera_params = flat_camera_params.reshape(n_cameras, n_cam_param)

        # update camera array with new positional data
        for index in range(len(new_camera_params)):
            print(index)
            port = index  # just to be explicit
            cam_vec = new_camera_params[index, :]
            self.cameras[port].extrinsics_from_vector(cam_vec)

    def bundle_adjust(self, point_data: PointData, output_path = None):
        # Original example taken from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

        camera_params = self.get_extrinsic_params()
        initial_param_estimate = np.hstack(
            (camera_params.ravel(), point_data.obj.ravel())
        )


        # get a snapshot of where things are at the start
        initial_xy_error = xy_reprojection_error(
            initial_param_estimate,
            self,
            point_data,
        )

        print(
            f"Prior to bundle adjustment, RMSE is: {rms_reproj_error(initial_xy_error)}"
        )
        
        # save out this snapshot if path provided
        if output_path is not None:
            diagnostic_data = ArrayDiagnosticData(point_data, initial_param_estimate,initial_xy_error, self)
            diagnostic_data.save(Path(output_path, "before_bund_adj.pkl"))


        least_sq_result = least_squares(
            xy_reprojection_error,
            initial_param_estimate,
            jac_sparsity=point_data.get_sparsity_pattern(),
            verbose=2,
            x_scale="jac",
            loss="linear",
            ftol=1e-8,
            method="trf",
            args=(
                self,
                point_data,
            ),
        )

        if output_path is not None:
            diagnostic_data = ArrayDiagnosticData(point_data, least_sq_result.x,least_sq_result.fun, self)
            diagnostic_data.save(Path(output_path, "after_bund_adj.pkl"))

        print(
            f"Following bundle adjustment, RMSE is: {rms_reproj_error(least_sq_result.fun)}"
        )
        return least_sq_result

    def optimize(self, point_data: PointData, output_path = None):
        """
        Currently, just run a simple bundle adjustment, noting the baseline reprojection errors before and after
        Use this as a way to characterize the quality of the camera configuration
        """

        for port, cam in self.cameras.items():
            print(f"Port {port} translation: {cam.translation.T}")

        least_sq_result = self.bundle_adjust(point_data)
        self.update_extrinsic_params(least_sq_result.x)

        # only examine best X% of fits...likely less impacted by distortion
        # point_data.filter(least_sq_result.fun, 0.5)
        # least_sq_result = self.bundle_adjust(point_data, ParamType.EXTRINSIC)
        # self.update_extrinsic_params(least_sq_result.x)

        for port, cam in self.cameras.items():
            print(f"Port {port} translation: {cam.translation.T}")
        # least_sq_result = self.bundle_adjust(point_data, ParamType.EXTRINSIC)
        # self.update_extrinsic_params(least_sq_result.x)

@dataclass
class ArrayDiagnosticData:
    point_data: PointData
    model_params: np.ndarray # the first argument of the residual function
    xy_reprojection_error: np.ndarray
    camera_array: CameraArray
    
    def save(self, output_path):
        with open(Path(output_path), 'wb') as file:
            pickle.dump(self,file)     



def xy_reprojection_error(
    current_param_estimates,
    camera_array,
    point_data,
):
    """
    current_param_estimates: the current iteration of the vector that was originally initialized for the x0 input of least squares
    """

    # Create one combined array primarily to make sure all calculations line up
    ## unpack the working estimates of the camera parameters (could be extr. or intr.)
    camera_params = current_param_estimates[
        : point_data.n_cameras * CAMERA_PARAM_COUNT
    ].reshape((point_data.n_cameras, CAMERA_PARAM_COUNT))

    ## similarly unpack the 3d point location estimates
    points_3d = current_param_estimates[
        point_data.n_cameras * CAMERA_PARAM_COUNT :
    ].reshape((point_data.n_obj_points, 3))

    ## create zero columns as placeholders for the reprojected 2d points
    rows = point_data.camera_indices.shape[0]
    blanks = np.zeros((rows, 2), dtype=np.float64)

    ## hstack all these arrays for ease of reference
    points_3d_and_2d = np.hstack(
        [
            np.array([point_data.camera_indices]).T,
            points_3d[point_data.obj_indices],
            point_data.img,
            blanks,
        ]
    )

    # iterate across cameras...while this injects a loop in the residual function
    # it should scale linearly with the number of cameras...a tradeoff for stable
    # and explicit calculations...
    for port, cam in camera_array.cameras.items():
        cam_points = np.where(point_data.camera_indices == port)
        object_points = points_3d_and_2d[cam_points][:, 1:4]

        cam_matrix = cam.camera_matrix
        rvec = camera_params[port][0:3]
        tvec = camera_params[port][3:6]
        distortion = cam.distortion[0]  # this may need some cleanup...

        # get the projection of the 2d points on the image plane; ignore the jacobian
        cam_proj_points, _jac = cv2.projectPoints(
            object_points.astype(np.float64), rvec, tvec, cam_matrix, distortion
        )

        points_3d_and_2d[cam_points, 6:8] = cam_proj_points[:, 0, :]

    points_proj = points_3d_and_2d[:, 6:8]

    # reshape the x,y reprojection error to a single vector
    return (points_proj - point_data.img).ravel()

def rms_reproj_error(xy_reproj_error):
        
    xy_reproj_error = xy_reproj_error.reshape(-1, 2)
    euclidean_distance_error = np.sqrt(np.sum(xy_reproj_error ** 2, axis=1))
    rmse = np.sqrt(np.mean(euclidean_distance_error**2))
    logging.info(f"Optimization run with {xy_reproj_error.shape[0]/2} image points")
    logging.info(f"RMSE of reprojection is {rmse}")
    return rmse

if __name__ == "__main__":
    from calicam.cameras.camera_array_builder import CameraArrayBuilder
    from calicam.calibration.bundle_adjustment.point_data import PointData, get_point_data

    repo = str(Path(__file__)).split("src")[0]

    session_directory = Path(repo, "sessions", "default_res_session")
    config_path = Path(session_directory, "config.toml")
    array_builder = CameraArrayBuilder(config_path)
    camera_array = array_builder.get_camera_array()

    # session_directory = Path(repo, "sessions", "iterative_adjustment")
    points_csv_path = Path(
        session_directory, "recording", "triangulated_points.csv"
    )

    point_data = get_point_data(points_csv_path)
    print(f"Optimizing initial camera array configuration ")
    # camera_array.optimize(point_data, output_path = points_csv_path.parent)
    camera_array.bundle_adjust(point_data, output_path = points_csv_path.parent)