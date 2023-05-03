import sys
import os

from pyxy3d.gui.calibration_wizard import launch_calibration_wizard

def launch():
    if len(sys.argv) == 1:
        print("No argument supplied")
        print(f"cwd: {os.getcwd()}")
        
    else:
        print(f"command line argument:{sys.argv}")
        print(f"cwd: {os.getcwd()}")

