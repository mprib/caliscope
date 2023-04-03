"""
A working file to hammer out the details of a check on whether the 
collected stereo data is sufficient to initialize an array... from there
the bundle adjustment can be used to refine it...
"""

#%%
board_counts = {(0, 1): 10, 
                (0, 2): 0, 
                (0, 3): 0, 
                (1, 2): 10, 
                (1, 3): 0, 
                (2, 3): 10}

                
                
#%%

missing_count_last_cycle = -1


missing_stereo_pairs = 

  
while len(_get_missing_stereopairs()) != missing_count_last_cycle:
            
    # prep the variable. if it doesn't go down, terminate
    missing_count_last_cycle = len(_get_missing_stereopairs())

    for pair in _get_missing_stereopairs():
             
        port_A = pair[0]
        port_C = pair[1]
    
        # get lists of all the estimiated stereopairs that might bridge across test_missing
        all_pairs_A_X = [pair for pair in estimated_stereopairs.keys() if pair[0]==port_A]
        all_pairs_X_C = [pair for pair in estimated_stereopairs.keys() if pair[1]==port_C]
   
        stereopair_A_C = None

        for pair_A_X in all_pairs_A_X:
            for pair_X_C in all_pairs_X_C:
                if pair_A_X[1] == pair_X_C[0]:
                    # A bridge can be formed!
                    stereopair_A_X = estimated_stereopairs[pair_A_X]
                    stereopair_X_C = estimated_stereopairs[pair_X_C]
                    possible_stereopair_A_C = get_bridged_stereopair(stereopair_A_X, stereopair_X_C)
                    if stereopair_A_C is None:
                        # current possibility is better than nothing
                        stereopair_A_C = possible_stereopair_A_C
                    else:
                        # check if it's better than what you have already
                        # if it is, then overwrite the old one
                        if stereopair_A_C.error_score > possible_stereopair_A_C.error_score:
                            stereopair_A_C = possible_stereopair_A_C

        if stereopair_A_C is not None:
            add_stereopair(stereopair_A_C)

if len(_get_missing_stereopairs()) > 0:
    raise ValueError("Insufficient stereopairs to allow array to be estimated")
