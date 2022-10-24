import sys
from pathlib import Path
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QTreeView
from PyQt6.QtGui import QStandardItem, QStandardItemModel, QFont, QColor, QImage
from PyQt6.QtCore import Qt
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
    def __init__(self, image, txt = ""):
        super().__init__()

        image = self.convert_cv_qt(image)
        # image = QImage(image)
        self.setEditable(False)
        self.setData(image, Qt.ItemDataRole.DecorationRole)
        self.setText(txt)

    def convert_cv_qt(self, cv_img):
            """Convert from an opencv image to QPixmap"""
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



class AppDemo(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.setWindowTitle("test window")
        self.resize(500, 700)

        treeView = QTreeView(self)
        treeView.setHeaderHidden(True)

        treeModel = QStandardItemModel()

        rootNode = treeModel.invisibleRootItem()

        self.setCentralWidget(treeView)

        # test_icon_path = r"C:\Users\Mac Prible\repos\learn-opencv\src\gui\icons\fmc_logo.png"

        # charuco = ImageItem(test_icon_path, "Charuco Board")
        charuco_header = StandardItem("Charuco", set_bold = True)
        rootNode.appendRow(charuco_header)

        edge_length_overide = session.config["charuco"]["square_size_overide"]
    
        charuco_img = ImageItem(session.charuco.board_img, f"Edge Length: {edge_length_overide} cm")
        charuco_header.appendRow(charuco_img)
        # for key, value in session.config["charuco"].items():
        #     if not key in ["dictionary", "aruco_scale"]:
        #         charuco.appendRow(StandardItem(f"{key}: {value}"))

        # rootNode.appendRow(charuco)

        cameras = StandardItem("Cameras")


        cam_rows = {} 
        for key, params  in session.config.items():
            if "cam" in key:    
                port = params["port"]
                cam_name = f"Camera {port}"
                # cam_rows[port] = StandardItem(cam_name)
                cam_rows[port] = params

        # a long way to deal with no sorting in python dictionaries
        cam_rows_sorted = {k:v for k, v in sorted(cam_rows.items(), key = lambda item:item[0])}

        for port, params in cam_rows_sorted.items():
           cam = StandardItem(f"Camera {port}")

           for key, value in params.items():
                item = StandardItem(key)
                item.appendRow(StandardItem(value))
                cam.appendRow(item)

           cameras.appendRow(cam)

        # print(cam_rows_sorted) 
        # print(cam_rows)
        print(f"Length of cam row dict is {len(cam_rows)}")

        rootNode.appendRow(cameras)

        treeView.setModel(treeModel)

        treeView.doubleClicked.connect(self.getValue)
        

    
    def getValue(self, val):
        print(val.data())
        print(val.row())
        print(val.column())

app = QApplication(sys.argv)

session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')

for key, params in session.config.items():
    print(key)
    print(params)

demo = AppDemo(session)

demo.show()

sys.exit(app.exec())

