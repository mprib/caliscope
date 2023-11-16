import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem
from pyxy3d.controller import Controller
import pyxy3d.logger
from collections import OrderedDict

logger = pyxy3d.logger.get(__name__)

class CameraDataDisplayWidget(QWidget):
    """
    This receives a dictionary that displays the characteristics of a given camera
    """
    def __init__(self, port:int, controller:Controller):
        super().__init__()
        self.port = port
        self.controller = controller
        self.tree = QTreeWidget()

        self.place_widgets()
        self.connect_widgets()

        # make sure you populate whatever is there on load
        self.controller.push_camera_data(self.port)

    def place_widgets(self):
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.tree)
        self.tree.setHeaderLabels(["Parameter", "Value"])


    def connect_widgets(self):
        self.controller.CameraDataUpdate.connect(self.update_tree)
        
    
    def update_tree(self, port, camera_display_dict):
        # port, camera_display_dict = port_camera_display_dict

        if port == self.port:
            self.tree.clear()
            # Adding top-level items
            self.add_items(None, self.tree, camera_display_dict)
            # Expand all items
            self.expand_all_items(self.tree)
        
        logger.info(camera_display_dict)        

    def add_items(self, parent, tree, data_dict):
        if parent is None:
            parent = tree

        for key, value in data_dict.items():
            if isinstance(value, OrderedDict):
                # For nested OrderedDict
                item = QTreeWidgetItem(parent, [key, ""])
                self.add_items(item, tree, value)
            else:
                # For direct key-value pairs
                QTreeWidgetItem(parent, [key, str(value)])

    def expand_all_items(self, tree):
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            self.expand_item(item)

    def expand_item(self, item):
        item.setExpanded(True)
        for i in range(item.childCount()):
            self.expand_item(item.child(i))

if __name__ == '__main__':
    from pyxy3d import __root__
    from pathlib import Path
    
    app = QApplication(sys.argv)


    camera_data = OrderedDict([
        ("size", (1920, 1080)),
        ("RMSE", 0.2),
        ("rotation_count", 3),
        ("intrinsic_parameters", OrderedDict([
            ("focal_length_x", 1000),
            ("focal_length_y", 1000),
            ("optical_center_x", 500),
            ("optical_center_y", 300)
        ])),
        ("distortion_coefficients", OrderedDict([
            ("radial_k1", .01),
            ("radial_k2", .02),
            ("radial_k3", .03),
            ("tangential_p1", .05),
            ("tangential_p2", .09)
        ]))
    ])

    test_path = Path(__root__, "tests", "sessions", "prerecorded_calibration")
    controller = Controller(test_path)
    
    ex = CameraDataDisplayWidget(port=0,controller=controller)
    controller.CameraDataUpdate.emit(0,camera_data)
    ex.show()
    sys.exit(app.exec())