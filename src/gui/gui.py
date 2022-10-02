from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, 
    QLabel, QLineEdit, QVBoxLayout, QGridLayout
)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

import sys

from qtpy import QT_API

class Window(QWidget):
    def __init__(self):
        super().__init__()
        # self.resize(300, 300)
        self.setWindowIcon(QIcon("fmc_logo.png"))
        self.setWindowTitle("FreeMoCap")
        self.setContentsMargins(20,20, 20, 20)  

        layout = QGridLayout() 
        self.setLayout(layout)

        title = QLabel("Login Form")
        title.setProperty("class", "heading")
        # title.setStyleSheet("font-size: 16px;")
        layout.addWidget(title, 0, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)

        
        label1 = QLabel("Username: ")
        label1.setProperty("class", "normal")
        self.input1 = QLineEdit()
        layout.addWidget(label1,1,0)
        layout.addWidget(self.input1, 1,1 )

        label2 = QLabel("Password: ")
        label2.setProperty("class", "normal")
        self.input2 = QLineEdit()
        layout.addWidget(label2,2,0)
        layout.addWidget(self.input2,2,1)


        button = QPushButton("Submit")
        button.setFixedWidth(50)
        button.clicked.connect(self.display)
        layout.addWidget(button, 3,1, Qt.AlignmentFlag.AlignRight )  


    def display(self): 
        print(self.input1.text())
        print(self.input2.text())



app = QApplication(sys.argv)

from pathlib import Path
with open(Path("src/gui/styleSheet.css")) as f:
    app.setStyleSheet(f.read())

# styleSheet = """
#     QWidget {
#         background-color: "green";
#         color: "white";
#     }

#     QLineEdit {
#         background-color: "white";
#     }

#     QPushButton {
#         font-size: 16px;
#     }
# """

# app.setStyleSheet(styleSheet)

window = Window()
window.show()
app.exec(app.exec())