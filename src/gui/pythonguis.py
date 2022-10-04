# Built following the tutorials that begin here: 
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/


# though generally frowned upon, these import all statements save a lot of
# typing and the names in PyQt6 don't class with other names
import sys

from PyQt6.QtGui import *
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                            QWidget)
from PyQt6.QtCore import QSize, Qt


# subclass QMainWindow into our window 
class  MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # initialize state of interface
        self.button_is_checked = True
        
        self.setWindowTitle("FreeMoCap")
        self.setWindowIcon(QIcon(r"src\gui\icons\fmc_logo.ico"))
        self.setFixedSize(QSize(400,300))

        label = QLabel("This is an awesome label")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        button = QPushButton("Press me")
        button.setCheckable(True)
        button.setChecked(self.button_is_checked)
        button.clicked.connect(self.button_toggled)        


        self.setCentralWidget(button)
    
    def button_clicked(self):
        print("Clicked")

    def button_toggled(self, checked):
        self.button_is_checked = checked
        self.setWindowTitle(f"The button is {self.button_is_checked}")

        print(self.button_is_checked)

app = QApplication(sys.argv)    # sys.argv allows passing in args from command line
 
window = MainWindow()   # must appear after the application is initalized
window.show()

app.exec() # 