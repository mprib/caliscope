
# %%

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

    def calibrate(self, board_threshold, charuco, charuco_inverted=False):
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
                read_success, frame = cap.read()

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

                    
                    # draw a box bounding each of the frames
                    frame = self.drawCharucoOutline(frame, charuco_corners, charuco_corner_ids)

##################### PROTOTYPING RAINBOX CALIBRATION
                    # draw a frame around the detected corners
                    # begin with a blank frame
                    
                                        
                    # just_corners =np.zeros(self.image_size[stream_name], dtype='uint8') 
                    # corner_lines = just_corners.copy()               

                    # # add circle at each of the detected corners to the blank image
                    # for crnr in charuco_corners:
                    #     cv.circle(just_corners, (round(crnr[0][0]), round(crnr[0][1])), 5, (255,255,255), -1)
                    


                    # for i in range(0, len(charuco_corners)-1):
                    #     crnr_x = round(charuco_corners[i][0][0])
                    #     crnr_y = round(charuco_corners[i][0][1])
                        
                    #     next_crnr_x = round(charuco_corners[i+1][0][0])
                    #     next_crnr_y = round(charuco_corners[i+1][0][1])
                        
                    #     cv.line(corner_lines, (crnr_x, crnr_y), (next_crnr_x,next_crnr_y), (255,255,255), 3)



                    # frame = corner_lines


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


    def drawCharucoOutline(self, frame, charuco_corners, charuco_ids):
        """
        Given a frame and the location of the charuco board corners within in,
        draw a polyline connecting the outer bounds of the detected corners
        """

        # can only draw a shape if there are more than 2 points
        if len(charuco_ids) > 3:
            
            # simplify the array you are working with
            # charuco_corners = charuco_corners.squeeze()
            # charuco_ids = charuco_ids.squeeze()

            # find min/max positions of corners            
            x_pos = charuco_corners[:,0,0]
            y_pos = charuco_corners[:,0,1]

            min_x = min(x_pos)
            max_x = max(x_pos)
            min_y = min(y_pos)
            max_y = max(y_pos)

            # limit corners to only those boundry cases
            outer_corners = []

            for x, y in charuco_corners[:,0,:]:

                if y == min_y:
                    outer_corners.append([0, round(x),round(y)])
                elif x == min_x:
                    outer_corners.append([1,round(x),round(y)] )                  
                elif y == max_y:
                    outer_corners.append([2,round(x),round(y)] )                  
                elif x == max_x:
                    outer_corners.append([3,round(x),round(y)] )                  

            
            # put corners in the correct order so that they don't criss-cross
            outer_corners.sort()

            # remove the sorting dimension
            outer_corners = np.delete(outer_corners, np.s_[0:1], axis=1)
            
            #required dataype for cv.polylines
            outer_corners = np.array(outer_corners, dtype=np.int32)

            return cv.polylines(frame, [outer_corners], True, (255,255,255), thickness=2)
        
        else:
            return frame



##############################      CLASS ENDS     ###################################
# Helper functions here primarily related to managing the charuco
# may consider organizing as a charuco class

def get_charuco():
    # create charuco for calibration
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

    # arguments: columns, rows, white space board?
    charuco_height_inch = 11 # inches
    charuco_width_inch = 8.5 # inches

    # convert to meters
    charuco_height = charuco_height_inch/39.37
    charuco_width = charuco_width_inch/39.37

    charuco_columns = 4
    charuco_rows = 5
    square_length = min([charuco_height/charuco_rows, 
                        charuco_width/charuco_columns]) 

    aruco_length = square_length * 0.9 

    board = cv.aruco.CharucoBoard_create(charuco_columns, charuco_rows, square_length, aruco_length, dictionary)

    return board


# %%

board = get_charuco()
corners = board.chessboardCorners

corners_x = corners[:,0]
corners_y = corners[:,1]

x_set = set(corners_x)
y_set = set(corners_y)

from collections import defaultdict

lines = defaultdict(list)

for x_line in x_set:
    for corner, x, y in zip(range(0, len(corners)), corners_x, corners_y):
        print(f"Corner: {corner}   x: {x}  y: {y}  x_line: {x_line}")

        if x == x_line:
            print("Added")
            lines[f"x_{x_line}"].append(corner)


for y_line in y_set:
    for corner, x, y in zip(range(0, len(corners)), corners_x, corners_y):
        if y == y_line:
            lines[f"y_{y_line}"].append(corner)


print(lines)


# %%
"""
For a given charuco board, returns the set of all corner pairs that are 
adjacent to each other and therefore eligible to be connect with a line 
when marking out the grid.    
"""
columns, rows = board.getChessboardSize()

corners_per_row = rows -1
corners_per_column = columns -1

corner_pairs = []

for corner in range(0, len(board.chessboardCorners)):
    # connect within rows
    # can link to next sequential corner, unless it is the last in the row
    if corner % (corners_per_row-1) == corners_per_row -2: # minus 2 due to 0 index
        pass
    elif corner % (corners_per_row-1) != corners_per_row -2:
        corner_pairs.append([corner, corner+1])

    # do the same for columns
    if corner % (corners_per_column-1) == corners_per_column -2: # minus 2 due to 0 index
        pass
    elif corner % (corners_per_column-1) != corners_per_column -2:
        corner_pairs.append([corner, corner + corners_per_row-1])


print(corner_pairs)



# %%

if __name__ == "__main__":
    feeds = CameraFeeds([0,1], ["Cam_1", "Cam_2"])
    feeds.calibrate(
        board_threshold=0.8,
        charuco = get_charuco(), 
        charuco_inverted=True)