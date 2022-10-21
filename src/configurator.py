# The main work of this class will be to maintain the consistency 
# between the session that is currently being run and the configuration
# that is stored in the session folder. This config.json file will 
# be  catchall for the camera settings and calibration. This 
# object will also initialize cameras that have not been configured
# yet
#%%
from collections import defaultdict
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from os.path import exists
#%%
from re import I
from threading import Thread
import toml
import cv2

# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cameras.camera import Camera

#%%


#%%
class Configurator:

    def __init__(self, session_folder):

        self.session_folder = session_folder
        self.config_path = str(Path(self.session_folder, "config.toml")) 

        # storage of parameters will be achieved via a nested dictionary
        # for ease of reference to recreate objects as needed and store a
        # variety of camera/chauco/whatever details
        if exists(self.config_path):
            self.config = toml.load(self.config_path)
            # self.dict = json.load(self.file)
            # self.dict = defaultdict({}, self.dict) 
        else:
            config_start = """
            path = 'test'
            """
            # self.config = toml.load(config_start)
            # self.dict = defaultdict()
            # self.dict["folder"] = self.session_folder            with open(self.config_path, 'w') as f:
                # dict_str = json.dumps(self.dict)
            with open(self.config_path, "a+") as f:
                print(config_start)
                parsed_config = toml.loads(config_start)
                toml.dump(parsed_config, f)

    def write_config(self):
        with open(self.config_path, 'w') as f:
            dict_str = json.dumps(self.dict)
            f.write(dict_str)

    def find_cameras(self, count):

        self.target_cam_count = count
        self.cameras = []

        def add_cam(port):
            try:
                print(f"Trying port {port}") 
                self.cameras.append(Camera(port))
                print(f"Success at port {port}")
            except:
                print(f"No camera at port {port}")
        

        with ThreadPoolExecutor() as executor:
            for i in range(0,10):

                executor.submit(add_cam, i )



    

test_session = r"C:\Users\Mac Prible\repos\learn-opencv\test_session"
config = Configurator(str(Path(test_session)))
# %%
config.find_cameras(3)
# config.update_cameras()
# config.write_config()

