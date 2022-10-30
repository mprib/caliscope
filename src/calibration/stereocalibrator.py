



    

    # session.adjust_resolutions()



    
    # build container for list of frames by port


    
    # now move across the corner_queue in a sequential way

    # all_frames = {}
    # for _ in range(corner_queue.qsize()):
    #     port_time_corners = corner_queue.get()
    #     port = port_time_corners["port"]
    #     frame_time = port_time_corners["frame_time"]
    #     frame_index = port_time_corners["frame_index"]
    #     frame_key = f"{port}_{frame_index}"

    #     all_frames[frame_key] = port_time_corners

#%%
    with open("frame_data.json","w") as f:
        json.dump(frame_data, f)