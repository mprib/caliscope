
#%%
import sys
from pathlib import Path
from tkinter.ttk import Treeview
import time
import toml
import cv2
from datetime import datetime
from os.path import exists
from PyQt6.QtCore import Qt, QSize, QAbstractItemModel
from PyQt6.QtGui import (QColor, QFont, QImage, QStandardItem,
                         QStandardItemModel)
from PyQt6.QtWidgets import QApplication, QMainWindow, QTreeView, QWidget, QSizePolicy

sys.path.insert(0,str(Path(__file__).parent.parent.parent))
from src.session import Session

class ConfigModel(QAbstractItemModel):
    def __init__(self, toml_path):
        super(ConfigModel, self).__init__()
        self.toml_path = toml_path
        self.load_config()

    def load_config(self):

        if exists(self.toml_path):
            print("Found previous config")
            with open(self.toml_path,"r") as f:
                self.config = toml.load(self.toml_path)
        else:
            print("Creating it")

            self.config = toml.loads("")
            self.config["CreationDate"] = datetime.now()
            with open(self.toml_path, "a") as f:
                toml.dump(self.config,f)

    def save_config(self):
        with open(self.toml_path, "w") as f:
           toml.dump(self.config,f)       

    
class StandardItem(QStandardItem):
    def __init__(self, txt="", font_size=12, set_bold=False, color=QColor(0,0,0)):
        super().__init__()

        fnt = QFont()
        fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(str(txt))

class ImageItem(QStandardItem):
    # needed to display charuco board
    def __init__(self, image, txt = ""):
        super().__init__()

        image = self.convert_cv_qt(image)
        # image = QImage(image)
        self.setEditable(False)
        self.setData(image, Qt.ItemDataRole.DecorationRole)
        self.setText(txt)

    def convert_cv_qt(self, cv_img):
            """Convert from an opencv image to QImage"""
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            charuco_QImage = QImage(rgb_image.data, 
                                    w, 
                                    h, 
                                    bytes_per_line, 
                                    QImage.Format.Format_RGB888)

            p = charuco_QImage.scaled(100,100,                                      
                                      Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)

            return p  #QPixmap.fromImage(p)



class ConfigTree(QWidget):
    def __init__(self, config_model):
        super().__init__()
        self.session = config_model 

        self.treeView = QTreeView(self)
        self.treeView.setHeaderHidden(True)

        self.treeModel = config_model
        self.treeView.setModel(self.treeModel)
        print("Works across modules")
        self.rootNode = self.treeModel.invisibleRootItem()
        # self.build_charuco()
        # self.build_camera_intrinisics()
    

    def build_camera_intrinisics(self):

        camera_intrinsics = StandardItem("Camera Intrinsics")
        self.rootNode.appendRow(camera_intrinsics)

        # build a sorted dictionary that can be used to construct a display
        # of intrinsic camera data
        cam_rows = {} 
        for key, params  in self.session.config.items():
            if "cam" in key:    
                port = params["port"]
                cam_rows[port] = params

        cam_rows_sorted = {k:v for k, v in sorted(cam_rows.items(), key = lambda item:item[0])}

        for port, params in cam_rows_sorted.items():
            cam = StandardItem(f"Camera {port}", set_bold=True)
            camera_intrinsics.appendRow(cam)

            cam.appendRow(StandardItem(f"Port: {port}"))

            res = params["resolution"]
            cam.appendRow(StandardItem(f"Resolution: {res}"))

            rotation = params["rotation_count"]
            if rotation == 0:
                rotation = "None"
            elif rotation == 1:
                rotation = "90 degrees"
            elif rotation in [-2,2]:
                rotation = "180 degrees"
            elif rotation in [-1,3]:
                rotation = "270 degrees"
            cam.appendRow(StandardItem(f"Rotation: {rotation}"))

    def build_charuco(self):

        charuco_header = StandardItem("Charuco", set_bold = True)
        self.rootNode.appendRow(charuco_header)
        # print(self.session.config)
        edge_length_overide = self.session.config["charuco"]["square_size_overide"]
    
        charuco_img = ImageItem(self.session.charuco.board_img, f"Edge Length: {edge_length_overide} cm")
        charuco_header.appendRow(charuco_img)



# if __name__ == "__main__":
#%%
if True:
    app = QApplication(sys.argv)

    # session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    toml_path = r"C:\Users\Mac Prible\repos\learn-opencv\test_session\config.toml"
    config_model = ConfigModel(toml_path)
#%%
    # for key, params in session.config.items():
    #     print(key)
    #     print(params)

    demo = ConfigTree(config_model)
    
    demo.show()

    time.sleep(1)

    # print(demo.treeModel.data())
    # demo.update()

    sys.exit(app.exec())

    