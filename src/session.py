
#%%

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os.path import exists
from pathlib import Path

import toml

sys.path.insert(0,str(Path(__file__).parent.parent))
# for p in sys.path:
#     print(p)

from src.calibration.charuco import Charuco
from src.cameras.camera import Camera

# config_path = str(Path(Path(__file__).parent.parent, "test_session", "config.toml"))
#%%

#%%
MAX_CAMERA_PORT_CHECK = 10

class Session:

    def __init__(self, directory):

        self.dir = str(directory)
        self.config_path = str(Path(self.dir, "config.toml"))
        self.load_config()

    def load_config(self):

        if exists(self.config_path):
            print("Found previous config")
            with open(self.config_path,"r") as f:
                self.config = toml.load(self.config_path)
        else:
            print("Creating it")

            self.config = toml.loads("")
            self.config["CreationDate"] = datetime.now()
            # self.config["charuco"] = ""
            # self.config["cameras"] = []
            with open(self.config_path, "a") as f:
                toml.dump(self.config,f)

        return self.config

    def update_config(self):
        with open(self.config_path, "w") as f:
           toml.dump(self.config,f)       


    def load_charuco(self):
        
        if "charuco" in self.config:
            print("Loading charuco from config")
            params = self.config["charuco"]
            
            # TOML doesn't seem to store None when dumping to file; adjust here
            if "square_size_overide" in self.config["charuco"]:
                sso = self.config["charuco"]["square_size_overide"]
            else:
                sso = None

            self.charuco = Charuco( columns = params["columns"],
                                    rows = params["rows"] ,
                                    board_height = params["board_height"],
                                    board_width = params["board_width"],
                                    dictionary = params["dictionary"],
                                    units = params["units"],
                                    aruco_scale = params["aruco_scale"],
                                    square_size_overide = sso,
                                    inverted = params["inverted"])
        else:
            print("Loading default charuco")
            self.charuco = Charuco(4,5,11,8.5)
            self.config["charuco"] = self.charuco.__dict__
            self.update_config() 

    def save_charuco(self):
        self.config["charuco"] = self.charuco.__dict__
        self.update_config()

    def find_cameras(self):
        # naming short and singular for brevity...cam[0] vs cameras[0]
        self.camera = {}

        def add_cam(port):
            try:
                print(f"Trying port {port}") 
                cam = Camera(port)
                print(f"Success at port {port}")
                self.camera[port] = cam
                self.save_camera(port)
            except:
                print(f"No camera at port {port}")

        with ThreadPoolExecutor() as executor:
            for i in range(0,MAX_CAMERA_PORT_CHECK):
                executor.submit(add_cam, i )


    def save_camera(self, port):
        cam = self.camera[port]
        params = {"port":cam.port,
                  "resolution": cam.resolution,
                  "rotation_count":cam.rotation_count,
                  "camera_matrix": cam.camera_matrix,
                  "distortion": cam.distortion}

        print(params)
        self.config["cam_"+str(port)] = params
        self.update_config()


#%%
# if __name__ == "__main__":
session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')

session.load_charuco()
# %%
session.charuco = Charuco(5,4,14,11, square_size_overide=None)
#%%

session.save_charuco()
# session.update_config()
# %%
session.find_cameras()
# %%

