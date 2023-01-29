from datetime import datetime
import time
from pathlib import Path

from calicam import __package_name__

BASE_FOLDER_NAME = f"{__package_name__}_data"
LOGS_INFO_AND_SETTINGS_FOLDER_NAME = "logs_info_and_settings"
LOG_FILE_FOLDER_NAME = "logs"

def get_base_folder_path():
    base_folder =  Path().home() / BASE_FOLDER_NAME
    base_folder.mkdir(exist_ok=True, parents=True)
    return base_folder

def get_log_file_path():
    log_file_path = get_base_folder_path() / LOGS_INFO_AND_SETTINGS_FOLDER_NAME / LOG_FILE_FOLDER_NAME / create_log_file_name()
    log_file_path.parent.mkdir(exist_ok=True, parents=True)
    return log_file_path

def create_log_file_name():
    return "log_" + get_iso6201_time_string() + ".log"


def get_gmt_offset_string():
    # from - https://stackoverflow.com/a/53860920/14662833
    gmt_offset_int = int(time.localtime().tm_gmtoff / 60 / 60)
    return f"{gmt_offset_int:+}"

def get_iso6201_time_string(timespec: str = "milliseconds", make_filename_friendly: bool = True):
    iso6201_timestamp = datetime.now().isoformat(timespec=timespec)
    gmt_offset_string = f"_gmt{get_gmt_offset_string()}"
    iso6201_timestamp_w_gmt = iso6201_timestamp + gmt_offset_string
    if make_filename_friendly:
        iso6201_timestamp_w_gmt = iso6201_timestamp_w_gmt.replace(":", "_")
        iso6201_timestamp_w_gmt = iso6201_timestamp_w_gmt.replace(".", "ms")
    return iso6201_timestamp_w_gmt
