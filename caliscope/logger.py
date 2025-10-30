import logging
import logging.handlers
import os
import sys

from PySide6 import QtCore

from caliscope import LOG_DIR


class QtHandler(logging.Handler):
    """
    Emits log records to a Qt signal, allowing them to be displayed in a GUI widget.
    """

    def __init__(self):
        super().__init__()
        qt_log_format = "%(name)s|%(message)s"
        self.setFormatter(logging.Formatter(qt_log_format))

    def emit(self, record):
        message = self.format(record)
        if message:
            XStream.stdout().write(f"{message}\n")


class XStream(QtCore.QObject):
    _stdout = None
    _stderr = None
    messageWritten = QtCore.Signal(str)

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg):
        if not self.signalsBlocked():
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        if not XStream._stdout:
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if not XStream._stderr:
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr


def setup_logging():
    """
    Configures the root logger for the entire application.
    This should be called only ONCE at the start of the application.
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Prevent adding handlers multiple times
    if root_logger.hasHandlers():
        return

    root_logger.setLevel(logging.INFO)
    log_format = "%(asctime)s | %(levelname)8s| %(name)3s| %(lineno)4d|  %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # 1. Add a rotating file handler
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "caliscope.log"
    log_file.touch(exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # 2. Add a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # 3. Add the Qt handler
    # skip in debug mode so you don't have to step through it
    if os.getenv("DEBUG") != "1":
        qt_handler = QtHandler()
        qt_handler.setLevel(logging.INFO)
        root_logger.addHandler(qt_handler)

    root_logger.info("Logging configured.")
