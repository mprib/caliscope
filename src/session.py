
#%%

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os.path import exists
from pathlib import Path

import toml

sys.path.insert(0,str(Path(__file__).parent.parent))

from src.calibration.charuco import Charuco
from src.cameras.camera import Camera

#%%
MAX_CAMERA_PORT_CHECK = 10

class Session:

    def __init__(self, directory):

        self.dir = str(directory)
        self.config_path = str(Path(self.dir, "config.toml"))

        # dictionary of Cameras, key = port
        self.camera = {}
        self.load_config()
        self.load_charuco()
        self.load_cameras()

    def load_config(self):

        if exists(self.config_path):
            print("Found previous config")
            with open(self.config_path,"r") as f:
                self.config = toml.load(self.config_path)
        else:
            print("Creating it")

            self.config = toml.loads("")
            self.config["CreationDate"] = datetime.now()
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

    def load_cameras(self):
        for key, item in self.config.items():
            if key.startswith("cam"):
                print(key, item)
                port = item["port"]
                self.camera[port] = Camera(port)

                cam =  self.camera[port] # trying to make a little more readable
                cam.resolution = item["resolution"]
                cam.rotation_count = item["rotation_count"]

                # if calibration done, then populate those
                if "error" in item.keys():
                    print(item["error"])
                    cam.error = item["error"] 
                    cam.camera_matrix = item["camera_matrix"]
                    cam.distortion = item["distortion"]


    def find_cameras(self):

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
                if i in self.camera.keys():
                    # don't try to connect to an already connected camera
                    pass
                else:
                    executor.submit(add_cam, i )

    def save_camera(self, port):
        cam = self.camera[port]
        params = {"port":cam.port,
                  "resolution": cam.resolution,
                  "rotation_count":cam.rotation_count,
                  "error": cam.error,
                  "camera_matrix": cam.camera_matrix,
                  "distortion": cam.distortion}

        print(params)
        self.config["cam_"+str(port)] = params
        self.update_config()


#%%
# if __name__ == "__main__":
session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')

# %%
session.charuco = Charuco(5,4,14,11, square_size_overide=None)
#%%

session.save_charuco()
# session.update_config()
# %%
# session.find_cameras()
session.load_cameras()
# %%

