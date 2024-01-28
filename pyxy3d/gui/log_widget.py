from caliscope.logger import get, XStream

from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QTextBrowser,
    QPushButton,
    QVBoxLayout,
)

from time import time
logger = get(__name__)


class LogWidget(QWidget):
    def __init__(self, message: str = None):
        super(LogWidget, self).__init__()
        self.setWindowTitle(message)
        self._console = LogMessageViewer(self)

        self.setWindowFlags(Qt.WindowType.WindowTitleHint)

        layout = QVBoxLayout()

        layout.addWidget(self._console)

        ## Verify widget working with a button to send a message
        if __name__ == "__main__":
            self._button = QPushButton(self)
            self._button.setText("Test Me")
            layout.addWidget(self._button)
            self._button.clicked.connect(test)

        self.setLayout(layout)
        XStream.stdout().messageWritten.connect(self._console.appendLogMessage)
        XStream.stderr().messageWritten.connect(self._console.appendLogMessage)


def test():
    logger.info(f"This is a test; It is {time()}")

class LogMessageViewer(QTextBrowser):
    def __init__(self, parent=None):
        super(LogMessageViewer, self).__init__(parent)
        self.setReadOnly(True)
        # self.setLineWrapMode(QtGui.QTextEdit.NoWrap)
        self.setEnabled(True)
        self.verticalScrollBar().setVisible(True)

    @Slot(str)
    def appendLogMessage(self, msg):
        horScrollBar = self.horizontalScrollBar()
        verScrollBar = self.verticalScrollBar()
        # scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value() <= 10

        verScrollBar.setValue(verScrollBar.maximum())  # Scrolls to the bottom
        horScrollBar.setValue(0)  # scroll to the left  
        self.insertPlainText(msg)

        # if scrollIsAtEnd:
        #     verScrollBar.setValue(verScrollBar.maximum())  # Scrolls to the bottom
        #     self.insertPlainText(msg)
        #     horScrollBar.setValue(0)  # scroll to the left  

if __name__ == "__main__":
    app = QApplication([])
    dlg = LogWidget("This is only a test")
    dlg.show()

    

    app.exec()
