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
