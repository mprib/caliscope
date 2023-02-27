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

class QtLogger(QDialog):
    def __init__( self, parent = None ):
        super(QtLogger, self).__init__(parent)

        self._console = LogMessageViewer(self)

        self._button  = QPushButton(self)
        self._button.setText('Test Me')
        self.vertical_scroll_bar = self._console.verticalScrollBar()

        layout = QVBoxLayout()
        layout.addWidget(self._console)
        layout.addWidget(self._button)

        self.setLayout(layout)
        XStream.stdout().messageWritten.connect( self._console.appendLogMessage)
        XStream.stderr().messageWritten.connect( self._console.appendLogMessage)

        self._button.clicked.connect(test)

class LogMessageViewer(QTextBrowser):

    def __init__(self, parent=None):
        super(LogMessageViewer,self).__init__(parent)
        self.setReadOnly(True)
        #self.setLineWrapMode(QtGui.QTextEdit.NoWrap)


    @QtCore.pyqtSlot(str)
    def appendLogMessage(self, msg):
        horScrollBar = self.horizontalScrollBar()
        verScrollBar = self.verticalScrollBar()
        scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value() <= 10

        self.insertPlainText(msg)

        if scrollIsAtEnd:
            verScrollBar.setValue(verScrollBar.maximum()) # Scrolls to the bottom
            horScrollBar.setValue(0) # scroll to the left
    
if __name__ == '__main__':
    app = QApplication([])
    dlg = QtLogger()
    dlg.show()
        
    app.exec()