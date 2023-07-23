import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
from pyxy3d import __root__
import pytest
import shutil
import cv2
from pathlib import Path
import time
from pyxy3d.trackers.hand_tracker import HandTracker
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.interface import PointPacket, FramePacket, SyncPacket
from pyxy3d.triangulate.sync_packet_triangulator import SyncPacketTriangulator
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d.configurator import Configurator
from pyxy3d.helper import copy_contents
from pyxy3d.trackers.tracker_enum import TrackerEnum

input_file = Path(__root__,"tests", "reference", "auto_rig_config_data", "")