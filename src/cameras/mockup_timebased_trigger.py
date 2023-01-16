

from time import perf_counter, perf_counter_ns, sleep
import numpy as np

fps = 3
# fps is two, which means that there are two fractional
# times at which we want frames: 0 and .5


milestones = []
for i in range(0, fps):
    print(i/fps)
    milestones.append(i/fps)

milestones = np.array(milestones)
print(milestones)


def wait_to_next_milestone():
    # time = perf_counter_ns()/(10**9) 
    time = perf_counter()
    fractional_time = time%1
    all_wait_times = milestones-fractional_time
    future_wait_times = all_wait_times[all_wait_times>0]
    
    if len(future_wait_times) ==0:
        # print("wait")
        return 1-fractional_time
    else:
        return future_wait_times[0]

while True:
    print(perf_counter())
    sleep(wait_to_next_milestone())
    print(wait_to_next_milestone())


# while True:
   
    
#     next_milestone = 1
#     print()