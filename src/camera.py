from logging import exception
import cv2 as cv
import time 
import numpy as np
class CameraFeeds():
    """
    Create a set of live OpenCV videoCapture devices. These can be then undergo
    individual calibration, stereocalibration, and collect data for 3d
    reconstruction 
    """

    def __init__(self, input_streams,stream_names):
        """
        Initialize input streams. One per camera. Video captures will be created
        for each one during active calibrations and data capture
        """
        
        #TODO: Figure out how to use exceptions
        # if len(stream_names)!=len(input_streams):
        # also, stream_names must all be unique    

        self.input_streams = input_streams
        self.stream_names = stream_names

    def calibrate_captures(self, board_threshold, charuco, charuco_inverted=False):
        """
        Charuco: a cv2 charuco board
        board_threshold: percent of board corners that must be represented to record
        
        """
        # prototyping calibration snapshots every second
        calibration_start = time.time()


        # build dictionary of all input streams
        self.captures = {}
        for stream_name, strm in zip(self.stream_names, self.input_streams):
            self.captures[stream_name] = cv.VideoCapture(strm)

        # and a place to record the image size
        self.image_size = {}
        for stream_name in self.stream_names:
            self.image_size[stream_name] = None

        # 
        # open the capture streams 
        while True:
            
            # for each stream
            for stream_name, cap in self.captures.items():
                # read in a frame
                _, frame = cap.read()

                # set the image size if uknown
                if self.image_size[stream_name] is None:
                    self.image_size[stream_name] = frame.shape

                # check for charuco corners in the image
                found_corner, charuco_corners, charuco_corner_ids = self.find_corners(
                    frame, 
                    charuco, 
                    charuco_inverted)


                # if charuco corners are detected
                if found_corner:

                    # draw them on the frame to visualize
                    frame = cv.aruco.drawDetectedCornersCharuco(
                        image = frame,
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_corner_ids,
                        cornerColor = (0,255,0))

                    

                    # draw a frame around the detected corners
                    blank_img =np.zeros(self.image_size[stream_name], dtype='uint8')                 

                    for crnr in charuco_corners:
                        cv.circle(blank_img, (round(crnr[0][0]), round(crnr[0][1])), 5, (255,0,0), 2)

                    frame = blank_img
                    # update the calibration corner data with this                    
                    # self.update_calibration_corners(charuco_corners, charuco_corner_ids)

                cv.imshow(stream_name, frame)

            if cv.waitKey(5) == 27: # ESC to stop   
                break

        self.destroy_captures()



    def destroy_captures(self):
        for nm, cap in self.captures.items():
            cap.release()
        cv.destroyAllWindows()
        self.captures = None

    def find_corners(self, frame, charuco, charuco_inverted):
        
        # invert the frame for detection if needed
        if charuco_inverted:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert to gray
            frame = ~frame  # invert
        

        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv.aruco.detectMarkers(
            frame, 
            charuco.dictionary)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 0:
            found, charuco_corners, charuco_corner_ids = cv.aruco.interpolateCornersCharuco(
                aruco_corners,
                aruco_ids,
                frame,
                charuco)
            
            if charuco_corners is not None:
                return True, charuco_corners, charuco_corner_ids
            else:
                return False, None, None

        else:
            return False, None, None



    def update_calibration_corners(self, corners, corner_ids):
        pass


        # return 



def get_charuco():
    # create charuco for calibration
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

    # arguments: columns, rows, white space board?
    charuco_border_inch = 0
    charuco_height_inch = 11 # inches
    charuco_width_inch = 8.5 # inches

    paper_height_inch = charuco_height_inch + charuco_border_inch
    paper_width_inch = charuco_width_inch + charuco_border_inch

    # convert to meters
    charuco_height = charuco_height_inch/39.37
    charuco_width = charuco_width_inch/39.37

    charuco_columns = 4
    charuco_rows = 5
    square_length = min([charuco_height/charuco_rows, 
                        charuco_width/charuco_columns]) 

    print(f"Square Length: {square_length}")

    aruco_length = square_length * 0.9 

    board = cv.aruco.CharucoBoard_create(charuco_columns, charuco_rows, square_length, aruco_length, dictionary)

    return board



if __name__ == "__main__":
    feeds = CameraFeeds([0,1], ["Cam_1", "Cam_2"])
    feeds.calibrate_captures(
        board_threshold=0.8,
        charuco = get_charuco(), 
        charuco_inverted=True)