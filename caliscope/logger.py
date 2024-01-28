# Construct a single logger that will be used throughout 
# Detail will be logged to a single file with INFO logged to the console

import logging
from PySide6 import QtCore
import sys
import os
from pathlib import Path
from caliscope import __log_dir__


# only one file handler accross package so all messages logged to one file

app_dir_file_handler = logging.FileHandler(Path(__log_dir__,'calibration.log'), "w+")
app_dir_file_handler.setLevel(logging.INFO)

file_log_format = " %(levelname)8s| %(name)30s| %(lineno)3d|  %(message)s"
file_formatter = logging.Formatter(file_log_format)
app_dir_file_handler.setFormatter(file_formatter)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

console_log_format =" %(levelname)8s| %(name)30s| %(lineno)3d|  %(message)s"
console_formatter = logging.Formatter(console_log_format)
console_handler.setFormatter(console_formatter)

# log_level_overides = {"caliscope.cameras.live_stream": logging.INFO}


class QtHandler(logging.Handler):
    """
    Adapted from discussion here: https://stackoverflow.com/questions/24469662/how-to-redirect-logger-output-into-pyqt-text-widget
    This handler will allow a QDialog box to pick up the logger output which may be useful for a  
    splash screen to show users that something is happening during big processing moments like:
    - loading/finding cameras

    - building / unbuilding synchronizer
    - performing stereocalibration
    """
    def __init__(self):
        logging.Handler.__init__(self)
        # qt_log_format =" %(levelname)s| %(name)s |%(message)s"
        qt_log_format =" %(name)s|%(message)s"
        qt_formatter = logging.Formatter(qt_log_format)
        self.setFormatter(qt_formatter)

    def emit(self, record):
        record = self.format(record)
        if record: XStream.stdout().write(f"{record} \n")


class XStream(QtCore.QObject):
    _stdout = None
    _stderr = None
    messageWritten = QtCore.Signal(str)
    def flush( self ):
        pass
    def fileno( self ):
        return -1
    def write( self, msg ):
        if ( not self.signalsBlocked() ):
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        if ( not XStream._stdout ):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if ( not XStream._stderr ):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr

def get(name): # as in __name__
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO) 
    
    logger.addHandler(app_dir_file_handler)
    logger.addHandler(console_handler)
 
    # avoid stepping through XStream object if in debug 
    if os.getenv("DEBUG") != '1':
        qt_handler = QtHandler()
        logger.addHandler(qt_handler)
    
    # qt_handler = QtHandler()
    # logger.addHandler(qt_handler)

    return logger

