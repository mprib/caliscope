# Construct a single logger that will be used throughout 
# Detail will be logged to a single file with INFO logged to the console

import logging
import logging.config

DEFAULT_LOGGING = {"version": 1, "disable_existing_loggers": True}
logging.config.dictConfig(DEFAULT_LOGGING)

logging.basicConfig(filemode='w', force=True)
logger = logging.getLogger(__name__)

file_log_format = " %(levelname)8s [%(filename)20s:%(lineno)3d] %(message)s"
file_formatter = logging.Formatter(file_log_format)


file_handler = logging.FileHandler('calibration.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)


console_log_format =" %(levelname)8s [%(filename)s:%(lineno)d] %(message)s"

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_log_format)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get():
    return logger