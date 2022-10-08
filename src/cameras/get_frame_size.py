# copied from https://www.learnpythonwithrune.org/find-all-possible-webcam-resolutions-with-opencv-in-python/

# %%
import pandas as pd
import cv2
import time
import copy

# url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
# table = pd.read_html(url)[0]
# table.columns = table.columns.droplevel()

start = time.time()

# %%
# table.to_excel("resolutions.xlsx")
cap2 = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
# note, if making an additional capture, it seems that the resolution options
# get narrowed considerably

# resolutions = {}
# for index, row in table[["W", "H"]].iterrows():
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     resolutions[str(width)+"x"+str(height)] = "OK"
# print(resolutions)


def get_nearest_resolution(capture, test_width):
    """returns a tuple of (width, height) that is closest to the tested width"""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, test_width)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return((width, height)) # print(str(width) + "x" + str(height))

# %%

start = time.time()

min_resolution = get_nearest_resolution(cap, 0)
print(time.time()-start)
start = time.time()

max_resolution = get_nearest_resolution(cap, 10000)
print(time.time()-start)
start = time.time()

resolutions = {min_resolution, max_resolution}


min_width = min_resolution[0]
max_width = max_resolution[0]
step_size = int((max_width-min_width)/6) # the size of jump to make before checking on the resolution

for test_width in range(int(min_width + step_size), int(max_width - step_size), int(step_size)):
    resolutions.add(get_nearest_resolution(cap, test_width))
    print(get_nearest_resolution(cap, test_width))

print(resolutions)

print(time.time()-start)
start = time.time()
# %%
middle_res_width = (smallest_res[0] + largest_res[0])/2
# %%

get_nearest_resolution(cap2, middle_res_width)
# %%

resolutions = {smallest_res, largest_res}

# %%
import cv2


def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    start = time.time()
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    elapsed = time.time()-start
    print(f"Elapsed time to get details on is {elapsed}")
    return available_ports,working_ports
# %%



from threading import Thread

def get_source_details(src, src_list):
    """returns success, source id ,default frame possible frames"""
    start = time.time()
    camera = cv2.VideoCapture(src)
    if not camera.isOpened():
        elapsed = time.time()-start
        print(f"Port {src} is not working.")
        print(f"Elapsed time to get details on is {elapsed}")
        # return False, None, [0,0]

    else:

        
        is_reading, img = camera.read()
        w = camera.get(3)
        h = camera.get(4)
        if is_reading:
            print("Port %s is working and reads images (%s x %s)" %(src,h,w))
        else:
            print("Port %s for camera ( %s x %s) is present but does not reads." %(src,h,w))
        elapsed = time.time()-start
        src_list.append([src, [w,h]])
        # return True, src, [w,h]
        print(f"Elapsed time to get details on is {elapsed}")



# %%

print("Beginning video scan...")
global_start = time.time()
threads = [] 
available_cameras = []

for src in range(0,5):
    get_source_thread = Thread(target = get_source_details, args=(src, available_cameras ))
    get_source_thread.daemon = True
    threads.append(get_source_thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print(available_cameras)
global_elapsed = time.time()-global_start




print(f"Total Camera Discovery Time: {global_elapsed}")
# %%
