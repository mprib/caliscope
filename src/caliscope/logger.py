import logging
import logging.handlers
import sys

from caliscope import LOG_FILE_PATH, LOG_DIR


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

    # Redirect stderr and set up exception hook
    sys.stderr = StderrLogger()
    sys.excepthook = handle_exception

    root_logger.info("Logging configured.")
