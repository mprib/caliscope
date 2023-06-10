from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QScrollArea,
)
from PyQt6.QtCore import Qt
from pyxy3d.configurator import Configurator
from pyxy3d.cameras.camera import Camera

class SummaryWidget(QWidget):
    def __init__(self, camera:Camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.setLayout(QVBoxLayout())
        self.place_widgets()

    # def clear_layout(self):
    #     while self.layout().count():
    #         child = self.layout().takeAt(0)
    #         if child.widget():
    #             child.widget().deleteLater()

    def place_widgets(self):
        # self.clear_layout()

        if self.camera.error is None:
            label = QLabel("Need to collect data...")
            self.layout().addWidget(label)
        else:
            # Create the first table
            scroll1 = QScrollArea()
            table1 = QTableWidget(3, 1)
            table1.setHorizontalHeaderLabels([""])
            table1.setVerticalHeaderLabels(["Grid Count", "Resolution", "RMSE"])
            table1.setItem(0, 0, QTableWidgetItem(str(self.camera.grid_count)))
            table1.setItem(
                1,
                0,
                QTableWidgetItem(f"{self.camera.size[0]} x {self.camera.size[1]}"),
            )
            table1.setItem(2, 0, QTableWidgetItem(str(round(self.camera.error, 3))))
            table1.horizontalHeader().setStretchLastSection(True)
            table1.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
            scroll1.setWidget(table1)
            scroll1.setWidgetResizable(True)
            table1.setColumnWidth(0, 50)
            for i in range(3):
                table1.item(i, 0).setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )

            # Create the second table
            scroll2 = QScrollArea()
            table2 = QTableWidget(2, 2)
            table2.setHorizontalHeaderLabels(["X", "Y"])
            table2.setVerticalHeaderLabels(["Focal Length", "Optical Center"])
            table2.setItem(0, 0, QTableWidgetItem(str(int(self.camera.matrix[0][0]))))
            table2.setItem(0, 1, QTableWidgetItem(str(int(self.camera.matrix[0][0]))))
            table2.setItem(1, 0, QTableWidgetItem(str(int(self.camera.matrix[0][2]))))
            table2.setItem(1, 1, QTableWidgetItem(str(int(self.camera.matrix[1][2]))))
            table2.horizontalHeader().setStretchLastSection(True)
            table2.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
            scroll2.setWidget(table2)
            scroll2.setWidgetResizable(True)
            for j in range(2):
                table2.setColumnWidth(j, 50)  # Adjust this value as necessary
            for i in range(2):
                for j in range(2):
                    table2.item(i, j).setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                    )

            # Create the third table
            scroll3 = QScrollArea()
            table3 = QTableWidget(5, 1)
            table3.setHorizontalHeaderLabels([""])
            table3.setVerticalHeaderLabels(
                [
                    "Radial 1",
                    "Radial 2",
                    "Tangential 1",
                    "Tangential 2",
                    "Radial 3",
                ]
            )
            for i in range(5):
                table3.setItem(
                    i, 0, QTableWidgetItem(str(round(self.camera.distortions[i], 3)))
                )
            table3.horizontalHeader().setStretchLastSection(True)
            table3.setColumnWidth(0, 50)  # Adjust this value as necessary
            table3.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
            scroll3.setWidget(table3)
            scroll3.setWidgetResizable(True)

            self.layout().addWidget(QLabel(""))
            self.layout().addWidget(scroll1)
            self.layout().addWidget(QLabel("Camera Intrinsics"))
            self.layout().addWidget(scroll2)
            self.layout().addWidget(QLabel("Camera Distortions"))
            self.layout().addWidget(scroll3)
            for i in range(5):
                table3.item(i, 0).setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )


            # make sure that the values don't get edited
            table1.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) 
            table2.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) 
            table3.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) 
            
