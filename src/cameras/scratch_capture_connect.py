

import cv2

test_port = 0

# https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
methods = [
    cv2.CAP_ANY, # success: 0
    cv2.CAP_VFW,
    cv2.CAP_V4L,
    cv2.CAP_V4L2,
    cv2.CAP_FIREWIRE,
    cv2.CAP_FIREWARE,
    cv2.CAP_IEEE1394,
    cv2.CAP_DC1394,
    cv2.CAP_CMU1394,
    cv2.CAP_QT,
    cv2.CAP_UNICAP,
    cv2.CAP_DSHOW, # success: 700
    cv2.CAP_PVAPI,
    cv2.CAP_OPENNI,
    cv2.CAP_OPENNI_ASUS,
    cv2.CAP_ANDROID,
    cv2.CAP_XIAPI,
    cv2.CAP_AVFOUNDATION,
    cv2.CAP_GIGANETIX,
    cv2.CAP_MSMF, # success: 1400
    cv2.CAP_WINRT,
    cv2.CAP_INTELPERC,
    cv2.CAP_OPENNI2,
    cv2.CAP_OPENNI2_ASUS,
    cv2.CAP_GPHOTO2,
    cv2.CAP_GSTREAMER,
    cv2.CAP_FFMPEG,
    cv2.CAP_IMAGES,
    cv2.CAP_ARAVIS,
    cv2.CAP_OPENCV_MJPEG,
    cv2.CAP_INTEL_MFX,
    cv2.CAP_XINE,
]

test_ports = [0,1,2]
successful_methods= {p:[] for p in test_ports}


for port in test_ports:
    for method in methods:
        print(f"Attempting to connect with {method}")
        test_Capture = cv2.VideoCapture(port, method)
        print("Connection successful")
        print("Attempting to read from capture")
        
        success, frame = test_Capture.read()
        
        if success: 
            cv2.imshow(str(method), frame)
            print(f"Success with method {method}")
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
            successful_methods[port].append(method)
        else:
            print(f"Failed to read frame using method {method}")

        
print(f"Successful Methods: {successful_methods}")