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
        
        label1 = QLabel("Username: ")
        self.input1 = QLineEdit()
        layout.addWidget(label1,0,0)
        layout.addWidget(self.input1, 0,1 )

        label2 = QLabel("Password: ")
        self.input2 = QLineEdit()
        layout.addWidget(label2,1,0)
        layout.addWidget(self.input2,1,1)


        button = QPushButton("Submit")
        button.setFixedWidth(50)
        button.clicked.connect(self.display)
        layout.addWidget(button, 2,1, Qt.AlignmentFlag.AlignRight )  


    def display(self): 
        print(self.input1.text())
        print(self.input2.text())



app = QApplication(sys.argv)


styleSheet = """
    QWidget {
        background-color: "green";
        color: "white";
    }

    QLineEdit {
        background-color: "white";
    }

    QPushButton {
        font-size: 16px;
    }
"""

# app.setStyleSheet(styleSheet)


window.show()
app.exec(app.exec())