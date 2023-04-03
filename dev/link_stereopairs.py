"""
A working file to hammer out the details of a check on whether the 
collected stereo data is sufficient to initialize an array... from there
the bundle adjustment can be used to refine it...
"""

#%%
# board_counts = {(0, 1): 10, 
#                 (0, 2): 0, 
#                 (0, 3): 0, 
#                 (1, 2): 10, 
#                 (1, 3): 0, 
#                 (2, 3): 10}
                
board_counts = {(0, 1): 0, (0, 2): 1, (0, 3): 1, (1, 2): 11, (1, 3): 1, (2, 3): 11}
min_threshold = 0

def get_empty_pairs(board_counts, min_threshold):
    empty_pairs = [key for key, value in board_counts.items() if value ==min_threshold]
    return empty_pairs



def possible_to_initialize_array(board_counts, min_threshold)-> bool:
# if progress was made on gap filling last time, try again 
# note use of walrus operator (:=). Not typically used  but it works here
    board_counts = board_counts.copy()
    flipped_board_counts = {tuple(reversed(key)): value for key, value in board_counts.items()}
    board_counts.update(flipped_board_counts)

    empty_count_last_cycle = -1

    while len(empty_pairs := get_empty_pairs(board_counts, min_threshold)) != empty_count_last_cycle:
        print("going through loop")
        # prep the variable. if it doesn't go down, terminate
        empty_count_last_cycle = len(empty_pairs)

        for pair in empty_pairs:
             
            port_A = pair[0]
            port_C = pair[1]

            all_pairs_A_X = [pair for pair in board_counts.keys() if pair[0]==port_A]
            all_pairs_X_C = [pair for pair in board_counts.keys() if pair[1]==port_C]
   
            board_count_A_C = None

            for pair_A_X in all_pairs_A_X:
                for pair_X_C in all_pairs_X_C:
                    if pair_A_X[1] == pair_X_C[0]:
                        # A bridge can be formed!
                        board_count_A_X = board_counts[pair_A_X]
                        board_count_X_C = board_counts[pair_X_C]
                    
                        if board_count_A_X > min_threshold and board_count_X_C > min_threshold:
                            board_count_A_C = min(board_count_A_X,board_count_X_C) 
                            board_counts[pair] = board_count_A_C
    
    if len(empty_pairs) > 0:
        is_possible = False
    else:
        is_possible = True

    return is_possible

possible_to_initialize_array(board_counts,min_threshold)



# %%
