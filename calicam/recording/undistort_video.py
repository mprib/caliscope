# this is likely just going to be a set of functions to save out a version of videos that has been undistorted.

from calicam.cameras.camera_array import CameraArray, CameraArrayBuilder, CameraData
from pathlib import Path
import cv2
from threading import Thread

def undistort_directory(camera_array: CameraArray, video_directory: Path):
    """provided with a camera array object and a directory containing
    raw videos, this function will create new versions of the video
    titled 'undistorted' that will be put in the same video directory"""

    for port, camera in camera_array.cameras.items():
        print(f"Port: {port}")
        thread = Thread(target=undistort_file, args=[camera, video_directory,], daemon=False)
        thread.start()
        
def undistort_file(camera: CameraData, video_directory, fps=30):
    read_path = str(Path(video_directory, f"port_{camera.port}.mp4"))
    capture = cv2.VideoCapture(read_path)

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    frame_size = camera.resolution
    write_path = str(Path(video_directory, "undistorted", f"port_{camera.port}.mp4"))
    writer = cv2.VideoWriter(write_path, fourcc,fps, frame_size )
            
    while True:
        success, raw_frame = capture.read()

        if success:
            undistorted_frame = cv2.undistort(
                raw_frame, camera.camera_matrix, camera.distortion
            )
            print(f"Writing frame at port {camera.port}")
            writer.write(undistorted_frame) 
        else:
            break 
    # must release writer to finalize file save after EOF reached
    print(f"Releasing writer at port {camera.port}")
    writer.release()

if __name__ == "__main__":

    repo = str(Path(__file__)).split("src")[0]
    config_path = Path(repo, "sessions", "iterative_adjustment", "config.toml")
    array_builder = CameraArrayBuilder(config_path)
    camera_array = array_builder.get_camera_array()

    video_directory = Path(repo, "sessions", "iterative_adjustment", "recording")

    undistort_directory(camera_array, video_directory)
    
    # undistort_file(camera_array.cameras[0],video_directory)
