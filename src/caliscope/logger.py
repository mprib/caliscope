import logging
import logging.handlers
import os
import sys

from caliscope import LOG_FILE_PATH, LOG_DIR


# Qt integration is optional -- only available when PySide6 is installed.
# Declare at module level so the type is visible regardless of import success.
qt_handler_instance: logging.Handler | None = None

try:
    from PySide6 import QtCore

    class LogEmitter(QtCore.QObject):
        """A simple QObject that holds the signal for the QtHandler."""

        message_written = QtCore.Signal(str)

    class QtHandler(logging.Handler):
        """A logging handler that emits log records via a Qt signal.

        Uses a LogEmitter instance (composition) to avoid method name clashes
        between logging.Handler.emit() and QObject signal emission.
        """

        def __init__(self):
            super().__init__()
            self.emitter = LogEmitter()

        def emit(self, record):
            message = self.format(record)
            if message:
                self.emitter.message_written.emit(message + "\n")

    qt_handler_instance = QtHandler()

except ImportError:
    pass


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

    with LOG_FILE_PATH.open("a") as f:
        f.write("Rotating Log File Handler Setting Up....")

    file_handler = logging.handlers.RotatingFileHandler(
        filename=LOG_FILE_PATH, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # 2. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # 3. Qt Handler (only when PySide6 available)
    # Don't step through if you are just debugging
    if qt_handler_instance is not None and os.getenv("DEBUG") != "1":
        qt_handler_instance.setLevel(logging.INFO)
        qt_format = "%(name)s: %(message)s"
        qt_formatter = logging.Formatter(qt_format)
        qt_handler_instance.setFormatter(qt_formatter)
        root_logger.addHandler(qt_handler_instance)

    # Redirect stderr and set up exception hook
    sys.stderr = StderrLogger()
    sys.excepthook = handle_exception

    root_logger.info("Logging configured.")
