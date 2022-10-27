# This widget is the primary functional unit of the motion capture. It
# establishes the connection with the video source and manages the thread
# that reads in frames.

from threading import Thread
import cv2
import time
import sys
import mediapipe as mp
import numpy as np
# Append main repo to top of path to allow import of backend
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera
from src.calibration.mono_calibrator import MonoCalibrator
from src.calibration.charuco import Charuco


class RealTimeDevice:
    def __init__(self, cam):
        # camera to be managed is the primary initiating component
        self.cam = cam 

        # self.cam.rotation_count = 0 
        
        # Start the thread to read frames from the video stream
        self.cap_thread = Thread(target=self.roll_camera, args=( ), daemon=True)
        self.cap_thread.start()
        self.frame_name = "Cam"+str(cam.port)
        
        # initialize time trackers for actual FPS determination
        self.avg_delta_time = None

        # Mediapipe hand detection infrastructure
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils 
        self.show_mediapipe = False

        # don't add anything special at the start
        self.charuco_being_tracked = False
        self.collect_charuco_corners = False
        self.undistort  = False 


    def get_FPS_actual(self):
        """set the actual frame rate; called within roll_camera()"""
        self.delta_time = time.time() - self.start_time
        self.start_time = time.time()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time

        # folding in current frame rate to trailing average to smooth out
        self.avg_delta_time = 0.95*self.avg_delta_time + 0.05*self.delta_time
        self.previous_time = self.start_time

        return 1/self.avg_delta_time

    def apply_rotation(self):

        if self.cam.rotation_count == 0:
            pass
        elif self.cam.rotation_count in [1, -3]:
            self._working_frame = cv2.rotate(self._working_frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.cam.rotation_count in [2,-2]:
            self._working_frame = cv2.rotate(self._working_frame, cv2.ROTATE_180)
        elif self.cam.rotation_count in [-1, 3]:
            self._working_frame = cv2.rotate(self._working_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)



    def run_mediapipe_hands(self):

         # Only calculate mediapipe if going to display it
        if self.show_mediapipe:
            frame_RGB  = cv2.cvtColor(self._working_frame, cv2.COLOR_BGR2RGB)
            self.hand_results = self.hands.process(frame_RGB)
        
            # draw hand dots and lines
            if self.hand_results.multi_hand_landmarks:
                for handLms in self.hand_results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(self._working_frame, handLms, self.mpHands.HAND_CONNECTIONS)

    def roll_camera(self):
        """
        Worker function that is spun up by Thread. Reads in a working frame, 
        calls various frame processing methods on it, and updates the exposed 
        frame

        """
        self.start_time = time.time() # used to get initial delta_t for FPS
        while True:
            self.cam.is_rolling = True

            if self.cam.capture.isOpened(): # note this line is truly necessary otherwise error upon closing capture
                # read in working frame
                self.status, self._working_frame = self.cam.capture.read()

                # REAL TIME OVERLAYS ON self._working_frame
                self.run_mediapipe_hands()
                self.process_charuco()
                self.apply_undistortion()

                self.apply_rotation() # must apply rotation at end...otherwise mismatch in frame / grid history dimensions

                # update frame that is emitted to GUI
                self.frame = self._working_frame.copy()

                # Rate of calling recalc must be limited by this loop
                self.FPS_actual = self.get_FPS_actual()

                # Stop thread if camera pulls trigger
                if self.cam.stop_rolling_trigger:
                    self.cam.is_rolling = False
                    break

    def change_resolution(self, res):
        self.cam.stop_rolling() # will trigger running capture thread to end
        
        # if the display isn't up and running this may error out (as when trying
        # to initialize the resolution to a non-default value)
        try:
            blank_image = np.zeros(self.frame.shape, dtype=np.uint8)
            self.frame = blank_image
        except:
            pass
        # pretty sure I can delete this next part since it is included in cam.stop_rolling()        
        # while self.cam.is_rolling:  # wait for everythong to catch up
        #     time.sleep(.01)

        self.FPS_actual = 0
        self.avg_delta_time = None

        # reconnecting a few times without disconnnect sometimes crashed python
        self.cam.disconnect()
        self.cam.connect()

        self.cam.resolution = res
        # if self.int_calib:
        try:
            self.mono_cal.initialize_grid_history()
        except:
            pass
        # test of commenting this out...may not longer be necessary
        # self.cap_thread.join()

        # Spin up the thread again now that resolution is changed
        self.cap_thread = Thread(target=self.roll_camera, args=( ), daemon=True)
        self.cap_thread.start()

    def toggle_mediapipe(self):
        self.show_mediapipe = not self.show_mediapipe
    
    def add_fps(self):
        """NOTE: this is used in code at bottom, not in external use """
        self.fps_text =  str(int(round(self.FPS_actual, 0))) 
        cv2.putText(self.frame, "FPS:" + self.fps_text, (10, 70),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 3)
    

    def assign_charuco(self, charuco):
        self.mono_cal = MonoCalibrator(self.cam, charuco)


    def process_charuco(self):
        """Heavy lifting from the charuco module. This method could involve just 
        displaying the identified corners on the frame, or adding them to 
        the list of corners for running a calibration. 
        
        The scope of the action depends on setting flags for:
        self.charuco_being_tracked
        self.collect_charuco_corners
        """
        if self.charuco_being_tracked:
            self.mono_cal.track_corners(self._working_frame, mirror=False)
            if self.collect_charuco_corners:
                self.mono_cal.collect_corners()

            self._working_frame = cv2.flip(self._working_frame,1)

            self.mono_cal.track_corners(self._working_frame, mirror=True)
            if self.collect_charuco_corners:
                self.mono_cal.collect_corners()

            self._working_frame = self.mono_cal.merged_grid_history()


    
    def apply_undistortion(self):

        if self.undistort == True: # and self.int_calib.is_calibrated:
            self._working_frame = cv2.undistort(self._working_frame,
                                                self.cam.camera_matrix,
                                                self.cam.distortion)
            

        

# Highlight module functionality. View a frame with mediapipe hands
# press "q" to quit
if __name__ == '__main__':
    ports = [0]
    
    cams = []
    for port in ports:
        print(f"Creating camera {port}")
        cams.append(Camera(port))

    charuco = Charuco(4,5,11,8.5,aruco_scale = .75, square_size_overide=.0525, inverted=True)

    real_time_devices = []
    for cam in cams:
        print(f"Creating Real Time Device for camera {cam.port}")
        rtd = RealTimeDevice(cam)
        rtd.assign_charuco(charuco)
        real_time_devices.append(rtd)
    

    while True:
        try:
            for rtd in real_time_devices:
                rtd.add_fps()
                cv2.imshow(str(rtd.frame_name +": 'q' to quit and attempt calibration"), rtd.frame)
                
        # bad reads until connection to src established
        except AttributeError:
            pass

        key = cv2.waitKey(1)

        # toggle mediapipe with 'm' 
        if key == ord('m'):
            print("Toggling Mediapipe")
            for rtd in real_time_devices:
                print(rtd.frame_name)
                rtd.toggle_mediapipe()
        
        if key == ord('r'):
            print("Rotate Frame CW")

            for rtd in real_time_devices:
                rtd.cam.rotate_CW()
                print(rtd.frame_name + " " + str(rtd.cam.rotation_count))
       
        if  key == ord('l'):
            print("Rotate Frame CCW")
                
            for rtd in real_time_devices:
                rtd.cam.rotate_CCW()
                print(rtd.frame_name + " " + str(rtd.cam.rotation_count))

        # Toggle charuco display
        if key == ord('c'):
            for rtd in real_time_devices:
                # rtd.assign_charuco(charuco)
                rtd.charuco_being_tracked = not rtd.charuco_being_tracked

        # Toggle charuco display
        if key == ord('C'):
            for rtd in real_time_devices:
                # rtd.assign_charuco(charuco)
                rtd.collect_charuco_corners = not rtd.collect_charuco_corners

        # Toggle undistortion
        if key == ord('d'):
            for rtd in real_time_devices:
                # rtd.assign_charuco(charuco)
                rtd.undistort = not rtd.undistort

        # 'q' to quit
        if key == ord('q'):
            for rtd in real_time_devices:
                try:
                    rtd.int_calib.calibrate()
                except:
                    pass
                rtd.cam.capture.release()
            cv2.destroyAllWindows()
            exit(0)

        if key == ord('v'):
            for rtd in real_time_devices:
                rtd.change_resolution((1280, 720))
