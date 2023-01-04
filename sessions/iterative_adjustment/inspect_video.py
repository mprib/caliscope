
import cv2
from pathlib import Path


video_path = r"C:\Users\Mac Prible\repos\calicam\sessions\iterative_adjustment\port_2.mp4"

# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(2)

while True: 
    success, frame = cap.read()
    cv2.imshow("Test Video", frame)
    
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break