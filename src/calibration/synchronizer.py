# This is a very much work in progress to sort through coding up the basic
# rule for frame syncing that I have in mind. Put all the frames in the first
# index. If one frame was read after the earliest frame in the next index,
# move it to the next index



import json
import logging

logging.basicConfig(filename="synchronizer.log", filemode = "w", level=logging.DEBUG)


with open("frame_data.json",) as f:
    frame_data = json.load(f)



# blank_layer = [None for _ in range(len(ports))]

# lay down initial layer of frames
# first_layer = blank_layer

# get minimum value of frame_time for next layer
def earliest_next_frame(ports, current_port_frame_indices, frame_data):
    next_frame_times = []
    for port in ports:
        next_index = current_port_frame_indices[port] + 1
        # port_index_key = str(port) + "_" + str(next_index)
        next_frame_at = frame_data[str(port)][next_index]["frame_time"]
        next_frame_times.append(next_frame_at)

    return min(next_frame_times)

ports = [0,1,2]
synched_frames = []
current_port_frame_indices = [0 for _ in range(len(ports))]

for _ in range(100):

    cutoff_time = earliest_next_frame(ports, current_port_frame_indices, frame_data)
    next_layer = []

    for port in ports:
        port_frame_index = current_port_frame_indices[port]
        # frame_data = all_frames[str(port) + "_" + str(port_frame_index)]
        frame_time = frame_data[str(port)][port_frame_index]["frame_time"]
        placeholder = f"{port}_{port_frame_index}_{frame_time}"

        if frame_data[str(port)][port_frame_index]["frame_time"] < cutoff_time:
            #add the data and increment the index
            next_layer.append(placeholder)      
            current_port_frame_indices[port] +=1
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