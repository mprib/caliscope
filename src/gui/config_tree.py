import sys
from pathlib import Path
from tkinter.ttk import Treeview
import time
import cv2
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import (QColor, QFont, QImage, QStandardItem,
                         QStandardItemModel)
from PyQt6.QtWidgets import QApplication, QMainWindow, QTreeView, QWidget, QSizePolicy

sys.path.insert(0,str(Path(__file__).parent.parent.parent))
from src.session import Session


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
    def __init__(self, session):
        super().__init__()
        self.session = session 

        self.treeView = QTreeView(self)
        self.treeView.setHeaderHidden(True)

        self.treeView.setModel(self.treeModel)
        print("Works across modules")
        self.treeModel = QStandardItemModel()
        self.rootNode = self.treeModel.invisibleRootItem()
        self.build_charuco()
        self.build_camera_intrinisics()
    

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



if __name__ == "__main__":
    app = QApplication(sys.argv)

    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')

    for key, params in session.config.items():
        print(key)
        print(params)

    demo = ConfigTree(session)
    
    demo.show()

    time.sleep(1)

    print(demo.treeModel.data())
    # demo.update()

    sys.exit(app.exec())

    