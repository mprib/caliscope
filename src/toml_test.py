
#%%

from cv2 import detail_Estimator
import toml
from pathlib import Path
from os.path import exists
import os
import time
import sys


sys.path.insert(0,str(Path(__file__).parent.parent))
for p in sys.path:
    print(p)

from src.calibration.charuco import Charuco

config_path = str(Path(Path(__file__).parent.parent, "test_session", "config.toml"))
#%%

if exists(config_path):
    print("removing")
    os.remove(config_path)



#%%

class Session:

    def __init__(self, directory):

        
        self.dir = str(directory)
        self.config = self.get_config()

        self.load_config()

    def get_config(self):
        config_path = str(Path(self.dir, "config.toml"))

        if exists(config_path):
            print("Found it")
            with open(config_path,"r") as f:

                config = toml.load(config_path)
        else:
            print("Creating it")

            config = toml.loads("")
            config["SessionDate"] = "today"

            with open(config_path, "a") as f:
                toml.dump(config,f)

        return config
                
    def load_config(self):
        """If there are any cameras or a charuco in the config, then go ahead
        and create them""" 
        pass

    def get_charuco(self):
        
        try:
            params = self.config["charuco"]

            self.charuco = Charuco( columns = params["columns"],
                                    rows = params["rows"] ,
                                    board_height = params["board_height"],
                                    board_width = params["board_width"],
                                    dictionary = params["dictionary"],
                                    units = params["units"],
                                    aruco_scale = params["aruco_scale"],
                                    square_size_overide = params["square_size_overide"],
                                    inverted = params["inverted"])
        except:
            self.charuco = Charuco(4,5,11,8.5)
            params = self.charuco.__dict__
            

        
         
#%%
# if __name__ == "__main__":
session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')

