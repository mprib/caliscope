import logging
import logging.handlers
import sys
from logging.config import dictConfig
from typing import Optional



DEFAULT_LOGGING = {"version": 1, "disable_existing_loggers": False}


def get_logging_handlers(log_file_path: Optional[str] = ""):
    dictConfig(DEFAULT_LOGGING)

    default_formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)04d] [%(levelname)8s] [%(name)s] [%(funcName)s():%(lineno)s] [PID:%(process)d "
        "TID:%(thread)d] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(default_formatter)
    handlers = [console_handler]
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(default_formatter)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    return handlers


def configure_logging(log_file_path: Optional[str] = ""):
    if len(logging.getLogger().handlers) == 0:
        handlers = get_logging_handlers(log_file_path)
        logging.getLogger("").handlers.extend(handlers)
        logging.root.setLevel(logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.info(f"Added logging handlers: {handlers}")
    else:
        logger = logging.getLogger(__name__)
        logger.info("Logging already configured!")
