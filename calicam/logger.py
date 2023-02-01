# Construct a single logger that will be used throughout 
# Detail will be logged to a single file with INFO logged to the console

import logging
import os
from pathlib import Path
from calicam import __log_dir__

# only one file handler accross package so all messages logged to one file

file_handler = logging.FileHandler(Path(__log_dir__,'calibration.log'), "w+")
file_handler.setLevel(logging.DEBUG)

file_log_format = " %(levelname)8s| %(name)30s| %(lineno)3d|  %(message)s"
file_formatter = logging.Formatter(file_log_format)
file_handler.setFormatter(file_formatter)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

console_log_format =" %(levelname)8s| %(name)30s| %(lineno)3d|  %(message)s"
console_formatter = logging.Formatter(console_log_format)
console_handler.setFormatter(console_formatter)

log_level_overides = {"calicam.cameras.live_stream": logging.INFO}

def get(name): # as in __name__
    print(f"Creating logger for {name}")
    logger = logging.getLogger(name)

    if name in log_level_overides.keys():
        logger.setLevel(log_level_overides[name])
    else:
        logger.setLevel(logging.DEBUG) 

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

