"""
Repository layer: persistence gateways for domain objects.

Repositories handle load/save operations to TOML/CSV files. They are the
boundary between domain logic and storage, converting between domain objects
and their serialized representations.
"""

from caliscope.repositories.camera_array_repository import CameraArrayRepository
from caliscope.repositories.charuco_repository import CharucoRepository
from caliscope.repositories.project_settings_repository import ProjectSettingsRepository
from caliscope.repositories.point_data_bundle_repository import PointDataBundleRepository

__all__ = [
    "CameraArrayRepository",
    "CharucoRepository",
    "ProjectSettingsRepository",
    "PointDataBundleRepository",
]
