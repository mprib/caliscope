import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QTreeView
from PyQt6.QtGui import QStandardItem, QStandardItemModel, QFont, QColor

sys.path.insert(0,str(Path(__file__).parent.parent.parent))
from src.session import Session

class StandardItem(QStandardItem):
    def __init__(self, txt="", font_size=12, set_bold=False, color=QColor(0,0,0)):
        super().__init__()

        # fnt = QFont('Open Sans', font_size)
        # fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        # self.setFont(fnt)
        self.setText(str(txt))


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

        charuco = StandardItem("Charuco Board")
        for key, value in session.config["charuco"].items():
            if not key in ["dictionary", "aruco_scale"]:
                charuco.appendRow(StandardItem(f"{key}: {value}"))

        rootNode.appendRow(charuco)

        cameras = StandardItem("Cameras")


        cam_rows = {} 
        for key, params  in session.config.items():
            if "cam" in key:    
                port = params["port"]
                cam_name = f"Camera {port}"
                cam_rows[port] = StandardItem(cam_name)

        # a long way to deal with no sorting in python dictionaries
        cam_rows_sorted = {k:v for k, v in sorted(cam_rows.items(), key = lambda item:item[1])}

        for key, value in cam_rows_sorted.items():
           cameras.appendRow(value) 

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

