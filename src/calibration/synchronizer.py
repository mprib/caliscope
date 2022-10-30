# This is a very much work in progress to sort through coding up the basic
# rule for frame syncing that I have in mind. Put all the frames in the first
# index. If one frame was read after the earliest frame in the next index,
# move it to the next index

import json
import logging
import time
from tkinter import E

logging.basicConfig(filename="synchronizer.log", filemode = "w", level=logging.DEBUG)


with open("frame_data.json",) as f:
    frame_data = json.load(f)

# get minimum value of frame_time for next layer
def earliest_next_frame(wait_retry = 0.03, attempt=0):
    
    try:
        time_of_next_frames = []
        for port in ports:
            next_index = port_current_frame[port] + 1
            # port_index_key = str(port) + "_" + str(next_index)
            next_frame_time = frame_data[str(port)][next_index]["frame_time"]
            time_of_next_frames.append(next_frame_time)
        return min(time_of_next_frames)

    except IndexError:
        if attempt > 5:
            raise IndexError
        logging.error("Not enough new frames available. Waiting for more frames...")
        time.sleep(wait_retry)
        logging.debug("Reattempting to get earliest next frame")
        earliest_next_frame(attempt=attempt+1)


ports = [0,1,2]
synched_frames = []
port_current_frame = [0 for _ in range(len(ports))]

while True:

    try:
        cutoff_time = earliest_next_frame()
    except IndexError as e:
        print(e)
        break

    next_layer = []

    for port in ports:
        current_frame = port_current_frame[port]
        # frame_data = all_frames[str(port) + "_" + str(port_frame_index)]
        current_frame_data = frame_data[str(port)][current_frame]
        frame_time = current_frame_data["frame_time"]

        # placeholder here is where the actual corner data would go
        placeholder = f"{port}_{current_frame}_{frame_time}"

        if frame_time < cutoff_time:
            #add the data and increment the index
            next_layer.append(placeholder)      
            port_current_frame[port] +=1
        else:
            next_layer.append(None)

    synched_frames.append(next_layer)

for i in range(len(synched_frames)):
    logging.debug(synched_frames[i])

#TODO: a function that looks at the growing port list and assesses if it is 
# time to call `earliest_next_frame()`

# current_port_frame_indices = [0 for _ in range(len(ports))]

# for key, frame_data in frame_data.items():

#     logging.debug(frame_data)
    # 