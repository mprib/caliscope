import logging
import logging.handlers
import os
import sys
from pathlib import Path

from PySide6 import QtCore

# Assuming LOG_DIR is defined in your project's __init__.py or config
# from caliscope import LOG_DIR
# Using a placeholder for this example:
LOG_DIR = Path("./logs")


class StderrLogger:
    """
    A file-like object that redirects writes to a logger.
    """

    def __init__(self, logger_name="stderr"):
        self.logger = logging.getLogger(logger_name)

    def write(self, message):
        if message.strip():
            self.logger.error(message.strip())

    def flush(self):
        pass


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception hook to log unhandled exceptions.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))


# Step 1: Create a dedicated QObject for emitting signals.
class LogEmitter(QtCore.QObject):
    """
    A simple QObject that holds the signal for the QtHandler.
    """

    message_written = QtCore.Signal(str)


# Step 2: QtHandler inherits ONLY from logging.Handler
class QtHandler(logging.Handler):
    """
    A logging handler that emits log records via a Qt signal.
    It USES a LogEmitter instance (composition) to avoid method name clashes.
    """

    def __init__(self):
        super().__init__()
        # Create an instance of our signal emitter
        self.emitter = LogEmitter()

    def emit(self, record):
        """
        This is the standard logging method. It now safely calls the
        signal on the separate emitter object.
        """
        message = self.format(record)
        if message:
            # Emit the signal from our dedicated emitter instance
            self.emitter.message_written.emit(message + "\n")


# Global instance of the QtHandler so it can be accessed from the LogWidget
qt_handler_instance = QtHandler()


def setup_logging():
    """
    Configures the root logger for the entire application.
    """
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        return

    root_logger.setLevel(logging.INFO)
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-15s | %(lineno)4d | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # 1. File Handler
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "caliscope.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # 2. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # 3. Qt Handler
    if os.getenv("DEBUG") != "1":
        qt_handler_instance.setLevel(logging.INFO)
        qt_format = "%(name)s: %(message)s"
        qt_formatter = logging.Formatter(qt_format)
        qt_handler_instance.setFormatter(qt_formatter)
        root_logger.addHandler(qt_handler_instance)

    # Redirect stderr and set up exception hook
    sys.stderr = StderrLogger()
    sys.excepthook = handle_exception

    root_logger.info("Logging configured.")
