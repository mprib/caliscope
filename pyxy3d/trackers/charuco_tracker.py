# There may be a mixed functionality here...I'm not sure. Between the corner
# detector and the corner drawer...like, there will need to be something that
# accumulates a frame of corners to be drawn onto the displayed frame.

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import cv2
import numpy as np

import pyxy3d.calibration.draw_charuco
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.interface import PointPacket, Tracker, TrackerFactory

class CharucoTracker(Tracker):
    def __init__(self, charuco):

        # need camera to know resolution and to assign calibration parameters
        # to camera
        self.charuco = charuco
        self.board = charuco.board
        self.dictionary_object = self.charuco.dictionary_object

        # for subpixel corner correction
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        self.conv_size = (11, 11)  # Don't make this too large.

    def get_points(self, frame):
        """Will check for charuco corners in the frame, if it doesn't find any, 
        then it will look for corners in the mirror image of the frame"""

        # invert the frame for detection if needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        if self.charuco.inverted:
            gray = ~gray  # invert

        ids, img_loc = self.find_corners_single_frame(gray, mirror=False)

        if not ids.any():
            gray = cv2.flip(gray, 1)
            ids, img_loc = self.find_corners_single_frame(gray, mirror=True)
        
        obj_loc = self.get_obj_loc(ids) 
        point_packet = PointPacket(ids, img_loc, obj_loc)
        
        # if len(self.ids) > 0:
        #     print("wait")
        
        return point_packet
    
    def get_point_names(self) -> dict:
        pass

    def get_connected_points(self):
        return self.charuco.get_connected_points()

    def find_corners_single_frame(self,gray_frame, mirror):

        ids = np.array([])
        img_loc = np.array([])

        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(
            gray_frame, self.dictionary_object
        )

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) >3:
            (success, _img_loc, _ids,) = cv2.aruco.interpolateCornersCharuco(
                aruco_corners, aruco_ids, gray_frame, self.board
            )

            # This occasionally errors out...
            # only offers possible refinement so if it fails, just move along
            try:
                _img_loc = cv2.cornerSubPix(
                    gray_frame,
                    _img_loc,
                    self.conv_size,
                    (-1, -1),
                    self.criteria,
                )
            except:
                logger.debug("Sub pixel detection failed")

            if success:
                # assign to tracker
                ids = _ids[:,0]
                img_loc = _img_loc[:,0]

                # flip coordinates if mirrored image fed in
                frame_width = gray_frame.shape[1]  # used for flipping mirrored corners back
                if mirror:
                    img_loc[:, 0] = frame_width - img_loc[:, 0]
                

                
        return ids, img_loc

    def get_obj_loc(self, ids:np.ndarray):
        """Objective position of charuco corners in a board frame of reference"""
        # if self.ids == np.array([0]):
            # print("wait")
        if len(ids) > 0:
            return self.board.getChessboardCorners()[ids, :]
        else:
            return np.array([])

    # @staticmethod    
    def draw_instructions(self, point_id: int) ->dict:
        rules = {"radius":5,
                 "color":(0,0,220),
                 "thickness":3}
        return rules

class CharucoTrackerFactory(TrackerFactory):
    
    def __init__(self, charuco:Charuco):
        self.charuco_tracker = CharucoTracker(charuco)
   
   
    def get_tracker(self) -> Tracker:
        return self.charuco_tracker
    
    def get_unique_name(self) -> str:
        return "charuco"
    
     
if __name__ == "__main__":

    from pyxy3d.cameras.camera import Camera
    from pyxy3d.cameras.live_stream import LiveStream
    
    tracker = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    cam = Camera(1)

    print(f"Using Optimized Code?: {cv2.useOptimized()}")
    trackr = CharucoTracker(tracker)
    stream = LiveStream(cam,fps_target=10,charuco=tracker)
    stream._show_fps = True
        
    print("About to enter main loop")
    while True:

        # read_success, frame = cam.capture.read()
        frame_packet = stream.out_q.get()
        pyxy3d.calibration.draw_charuco.corners(frame_packet)

        cv2.imshow("Press 'q' to quit", frame_packet.frame)
        key = cv2.waitKey(1)

        # end capture when enough grids collected
        if key == ord("q"):
            cam.capture.release()
            cv2.destroyAllWindows()
            break
