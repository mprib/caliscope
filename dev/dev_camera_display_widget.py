import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem

class CameraDataDisplayWidget(QWidget):
    def __init__(self, camera_data):
        super().__init__()
        self.camera_data = camera_data
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Parameter", "Value"])

        # Adding top-level items
        self.add_items(None, tree, self.camera_data)

        # Expand all items
        self.expand_all_items(tree)

        layout.addWidget(tree)

        self.setWindowTitle("Camera Data Display")
        self.show()

    def add_items(self, parent, tree, data_dict):
        if parent is None:
            parent = tree

        for key, value in data_dict.items():
            if isinstance(value, dict):
                # For nested dictionaries
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
    app = QApplication(sys.argv)

    # Example camera data
    camera_data = {
        "size": "(1920, 1080)",
        "rotation_count": "3",
        "intrinsic_parameters": {
            "focal_length_x": "1000",
            "focal_length_y": "1000",
            "optical_center_x": "500",
            "optical_center_y": "500",
            "distortion_coefficients": {
                "radial_k1": "0.1",
                "radial_k2": "0.01",
                "radial_k3": "0.001",
                "tangential_p1": "0.005",
                "tangential_p2": "0.005"
            }
        },
        "rmse": "0.2"
    }

    ex = CameraDataDisplayWidget(camera_data)
    sys.exit(app.exec())
