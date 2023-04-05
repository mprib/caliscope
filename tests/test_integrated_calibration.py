
"""
It's occuring to me that this is not really the way to do it because I need
to sort out a way to make this work with recorded video, which will involve passing
recording streams into some things instead of live streams. 

This is a good exercise for me at this point prior to embarking on the next phase of things.
I can take my time and learn from my mistakes on this first iteration.
I do think that when I have tests ready I may bump things up to 0.1.0. The minor
update will be to signify that the initial calibration functionality is somewhat wrapped.

And then I move on to the next stage of things...I think I'm going to pop over to 
Obsidian to flesh out a path for the milestones of the project...

"""
import pytest
import os
from pathlib import Path

from pyxy3d.session import Session
from pyxy3d import __root__

sample_session_paths = [p for p in Path(__root__,"tests", "sessions").glob("*/") if p.is_dir() ]
    
print(sample_session_paths)

for session_path in sample_session_paths:
    session = Session(session_path)
    

    