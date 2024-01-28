import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem
from caliscope.controller import Controller
import caliscope.logger
from collections import OrderedDict

logger = caliscope.logger.get(__name__)

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
        self.controller.new_camera_data.connect(self.update_tree)
        
    
    def update_tree(self, port, camera_display_dict):
        # logger.info(f"Updating display tree for port {port} with camera data {camera_display_dict}")
        # port, camera_display_dict = port_camera_display_dict

        if port == self.port:
            self.tree.clear()
            # Adding top-level items
            self.add_items(None, self.tree, camera_display_dict)
            # Expand all items
            self.expand_all_items(self.tree)
        
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_column_widths()

    def adjust_column_widths(self):
        total_width = self.tree.width() - 2  # Subtracting 2 pixels for border
        self.tree.setColumnWidth(0, int(total_width * 0.7))  # 70% for parameter column
        self.tree.setColumnWidth(1, int(total_width * 0.3))  # 30% for value column


if __name__ == '__main__':
    from caliscope import __root__
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
    controller.config.dict["camera_count"] = 1
    ex = CameraDataDisplayWidget(port=0,controller=controller)
    controller.new_camera_data.emit(0,camera_data)
    ex.show()
    sys.exit(app.exec())
