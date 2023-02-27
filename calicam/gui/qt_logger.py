import sys
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QDialog, QApplication, QTextBrowser, QPushButton, QVBoxLayout
import logging

from calicam.session import Session
from pathlib import Path
from threading  import Thread

def test():
    def worker():
        session = Session(Path(r"C:\Users\Mac Prible\repos\calicam\sessions\laptop"))
        session.find_cameras()
    thread = Thread(target=worker, args=(), daemon=True)
    thread.start()

from calicam.logger import get, XStream
logger = get(__name__)

class LoggerPopUp(QDialog):
    def __init__( self, parent = None ):
        super(LoggerPopUp, self).__init__(parent)

        self._console = QTextBrowser(self)
        self._button  = QPushButton(self)
        self._button.setText('Test Me')

        layout = QVBoxLayout()
        layout.addWidget(self._console)
        layout.addWidget(self._button)
        self.setLayout(layout)

        XStream.stdout().messageWritten.connect( self._console.insertPlainText )
        XStream.stderr().messageWritten.connect( self._console.insertPlainText )

        self._button.clicked.connect(test)

if __name__ == '__main__':
    app = QApplication([])
    dlg = LoggerPopUp()
    dlg.show()
        
    app.exec()