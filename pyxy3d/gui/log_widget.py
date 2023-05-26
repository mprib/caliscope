
import logging
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QApplication
import sys
from threading import Thread
from time import time, sleep

class SignalHandler(logging.Handler, QObject):
    log_message = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_message.emit(msg)

class LogFileWatcher(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Start a thread that logs a message every second
        self.thread = Thread(target=self.random_write, args=(), daemon=True)
        self.thread.start()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.log_view = QTextEdit()
        self.log_view.setEnabled(False)
        self.layout.addWidget(self.log_view)
        self.setLayout(self.layout)

        # Create a logging handler that emits a signal when a message is logged
        self.handler = SignalHandler()
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.handler.log_message.connect(self.handle_message)

        # Create a logger and add the handler to it
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def random_write(self):
        while True:
            self.logger.info(f"Time is {time()}")
            sleep(1)

    def handle_message(self, message):
        self.log_view.append(message)  # append the log message to the QTextEdit

if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = LogFileWatcher()
    widget.show()

    sys.exit(app.exec())
