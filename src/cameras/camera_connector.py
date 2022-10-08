# A module to manage setup of live camera captures that are provided to the 
# video capture widget

#%%

import cv2
from threading import Thread
import time

CAM_COUNT = 4



def store_capture(src, cap_list):
    """
    adds to caplist a list of [scr, cv2.VideoCapture(src)] 
    for reference elsewhere
    """

    start = time.time()
    camera = cv2.VideoCapture(src)

    if not camera.isOpened():
        elapsed = time.time()-start
        print(f"Port {src} is not working.")
        print(f"Elapsed time to get details on is {elapsed}")

    else:
        
        # attempt to read a few frames to make sure there is an actual feed
        test_image_count = 5
        for i in range(0,test_image_count): 
             is_reading, img = camera.read()

        if is_reading:
            print(f"Port {src} is working and reading images")
            cap_list.append(camera)
        else:
            print("Port %s for camera ( %s x %s) is present but does not reads." %(src,h,w))
        elapsed = time.time()-start

        print(f"Elapsed time to get details on is {elapsed}")

# Process without Threading
#%%
captures = []

threads = []

for src in range(0,5):
    store_cap_thread = Thread(target = store_capture, args=(src, captures))
    store_cap_thread.daemon = True
    threads.append(store_cap_thread)

for thread in threads:
    thread.start()

# print capture list every 3 seconds

def print_caps(captures):
    while len(captures) < 4:
        print(captures)
        time.sleep(3)

print_thread = Thread(target = print_caps, args = (captures,))
print_thread.start()

for thread in threads:
    thread.join()

#%%
# Show Camera Feeds



# for i in range(0,4):
#     store_capture(i, captures)
#%% 
# Display Captures


#%%
# release captures

for cap in captures:
    cap.release()