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
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(0)

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

print(time.time()-start)
start = time.time()

smallest_res = get_nearest_resolution(cap2, 0)
print(smallest_res)
print(time.time()-start)
start = time.time()


largest_res = get_nearest_resolution(cap2, 10000)
print(largest_res)
print(time.time()-start)
