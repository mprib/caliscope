
import logging
import pprint

LOG_FILE = r"log\stereo_triangulator.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
from pathlib import Path
import pyqtgraph as pg
import pyqtgraph.opengl as gl

sys.path.insert(0,str(Path(__file__).parent.parent.parent))
from src.triangulate.visualization.camera_mesh import CameraMesh

class StereoVisualizer:
    
    def __init__(self, triangulator):
        self.triangulator = triangulator
        
        self.cam_A = triangulator.camera_A
        self.cam_B = triangulator.camera_B
 
        # create the overhead for display 
        self.app = pg.mkQApp("GLMeshItem Example")
        self.scene = gl.GLViewWidget()
        self.scene.show()
        self.scene.setWindowTitle('Camera Calibration')
        self.scene.setCameraPosition(distance=4)
        
        grid = gl.GLGridItem()
        grid.scale(1,1,1)

        self.scene.addItem(grid)
        self.scene.addItem(self.cam_A.mesh)

        pg.exec()


        
    def show(self):
        pass
          
if __name__ == '__main__':

    from src.triangulate.stereo_triangulator import StereoTriangulator
    
    sample_config_path  = str(Path(Path(__file__).parent, "sample_data", "config.toml"))
    trigr = StereoTriangulator(0,1,sample_config_path)    

    vizr = StereoVisualizer(trigr)