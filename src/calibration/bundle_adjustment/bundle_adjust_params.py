from pathlib import Path

# Going to start pulling in variables from the csv files that are saved out and
# attempt to get the data formatted in the same way that is required by the 
# sample code from scipy.org



# camera_params
# camera_params with shape (n_cameras, 9) contains initial estimates of parameters for all cameras. 
# First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), 
# next 3 components form a translation vector, then a focal distance and two distortion parameters.
# note that the distortion parameters only reflect the radial distortion (not the tangential) 



# points_3d
# points_3d with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.



# camera_id
# camera_id with shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.




# point_ind
# point_ind with shape (n_observations,) contains indices of points (from 0 to n_points - 1) involved in each observation.





# points_2d
# points_2d with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.

if __name__ == "__main__":
    repo = str(Path(__file__)).split("src")[0]

    config_file = Path(repo, "sessions", "iterative_adjustment", "config.toml")
    array_builder = CameraArrayBuilder(config_file)

    camera_array = array_builder.get_camera_array()
    
    print("pause")