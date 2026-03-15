"""Composite repository for all calibration target configurations.

Owns the `calibration/targets/` directory and provides role-based access
(intrinsic vs extrinsic) to different target types. Wraps existing persistence
functions and adds routing logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rtoml

from caliscope.persistence import PersistenceError
from caliscope.core.aruco_target import ArucoTarget
from caliscope.core.charuco import Charuco
from caliscope.core.chessboard import Chessboard

logger = logging.getLogger(__name__)

IntrinsicTargetType = Literal["charuco", "chessboard"]
ExtrinsicTargetType = Literal["charuco", "aruco"]


@dataclass(frozen=True, slots=True)
class TargetRouting:
    """Routing config: which target type serves each calibration role."""

    intrinsic_target_type: IntrinsicTargetType
    extrinsic_target_type: ExtrinsicTargetType
    extrinsic_charuco_same_as_intrinsic: bool


class CalibrationTargetsRepository:
    """Composite repository for all calibration target configurations.

    Owns the `calibration/targets/` directory. Wraps existing persistence
    functions for individual target types and adds routing logic via
    `config.toml`.

    File layout:
        calibration/targets/
            config.toml              # Routing (which type for each role)
            intrinsic_charuco.toml   # ChArUco config for intrinsic role
            extrinsic_charuco.toml   # ChArUco config for extrinsic role (when not same_as_intrinsic)
            chessboard.toml          # Chessboard config
            aruco_target.toml        # ArUco target config
    """

    def __init__(self, targets_dir: Path) -> None:
        """
        Args:
            targets_dir: Path to calibration/targets/ directory.
        """
        self._dir = targets_dir
        self._config_path = targets_dir / "config.toml"

    # -- Routing ----------------------------------------------------------

    def get_routing(self) -> TargetRouting:
        """Load routing config from config.toml.

        Returns default routing if file doesn't exist.
        """
        if not self._config_path.exists():
            return TargetRouting(
                intrinsic_target_type="charuco",
                extrinsic_target_type="charuco",
                extrinsic_charuco_same_as_intrinsic=True,
            )
        data = rtoml.load(self._config_path)
        return TargetRouting(
            intrinsic_target_type=data.get("intrinsic_target_type", "charuco"),
            extrinsic_target_type=data.get("extrinsic_target_type", "charuco"),
            extrinsic_charuco_same_as_intrinsic=data.get("extrinsic_charuco_same_as_intrinsic", True),
        )

    def save_routing(self, routing: TargetRouting) -> None:
        """Save routing config to config.toml."""
        self._dir.mkdir(parents=True, exist_ok=True)
        data = {
            "intrinsic_target_type": routing.intrinsic_target_type,
            "extrinsic_target_type": routing.extrinsic_target_type,
            "extrinsic_charuco_same_as_intrinsic": routing.extrinsic_charuco_same_as_intrinsic,
        }
        with open(self._config_path, "w") as f:
            rtoml.dump(data, f)

    @property
    def intrinsic_target_type(self) -> IntrinsicTargetType:
        """Current intrinsic target type (convenience)."""
        return self.get_routing().intrinsic_target_type

    @property
    def extrinsic_target_type(self) -> ExtrinsicTargetType:
        """Current extrinsic target type (convenience)."""
        return self.get_routing().extrinsic_target_type

    @property
    def extrinsic_charuco_same_as_intrinsic(self) -> bool:
        """Whether extrinsic charuco shares intrinsic config (convenience)."""
        return self.get_routing().extrinsic_charuco_same_as_intrinsic

    # -- Intrinsic Charuco ------------------------------------------------

    def load_intrinsic_charuco(self) -> Charuco:
        """Load charuco for intrinsic calibration.

        Reads from intrinsic_charuco.toml.
        Raises ValueError if file doesn't exist.
        """
        path = self._dir / "intrinsic_charuco.toml"
        try:
            return Charuco.from_toml(path)
        except PersistenceError as e:
            raise ValueError(f"Failed to load intrinsic charuco: {e}") from e

    def save_intrinsic_charuco(self, charuco: Charuco) -> None:
        """Save charuco for intrinsic calibration.

        Writes to intrinsic_charuco.toml.
        If extrinsic_charuco_same_as_intrinsic is true, this also
        affects what load_extrinsic_charuco() returns (it reads
        intrinsic_charuco.toml in that case).
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            charuco.to_toml(self._dir / "intrinsic_charuco.toml")
        except PersistenceError as e:
            raise ValueError(f"Failed to save intrinsic charuco: {e}") from e

    def intrinsic_charuco_exists(self) -> bool:
        """Check if intrinsic_charuco.toml exists."""
        return (self._dir / "intrinsic_charuco.toml").exists()

    # -- Extrinsic Charuco ------------------------------------------------

    def load_extrinsic_charuco(self) -> Charuco:
        """Load charuco for extrinsic calibration.

        If same_as_intrinsic is true, reads from intrinsic_charuco.toml.
        Otherwise reads from extrinsic_charuco.toml.
        Raises ValueError if the resolved file doesn't exist.
        """
        routing = self.get_routing()
        if routing.extrinsic_charuco_same_as_intrinsic:
            return self.load_intrinsic_charuco()
        else:
            path = self._dir / "extrinsic_charuco.toml"
            try:
                return Charuco.from_toml(path)
            except PersistenceError as e:
                raise ValueError(f"Failed to load extrinsic charuco: {e}") from e

    def save_extrinsic_charuco(self, charuco: Charuco) -> None:
        """Save charuco for extrinsic calibration.

        Always writes to extrinsic_charuco.toml.
        Should only be called when same_as_intrinsic is false.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            charuco.to_toml(self._dir / "extrinsic_charuco.toml")
        except PersistenceError as e:
            raise ValueError(f"Failed to save extrinsic charuco: {e}") from e

    def set_extrinsic_charuco_same_as_intrinsic(self, same: bool) -> None:
        """Toggle the same_as_intrinsic flag.

        When toggling from false -> true: no file copy needed (reads
        redirect to intrinsic_charuco.toml).

        When toggling from true -> false: copies intrinsic_charuco.toml
        to extrinsic_charuco.toml as a starting point (so the extrinsic
        panel opens with the same values the user was previously seeing).
        """
        routing = self.get_routing()
        if not same and routing.extrinsic_charuco_same_as_intrinsic:
            # Toggling OFF: copy intrinsic -> extrinsic as starting point
            intrinsic = self.load_intrinsic_charuco()
            intrinsic.to_toml(self._dir / "extrinsic_charuco.toml")
        new_routing = TargetRouting(
            intrinsic_target_type=routing.intrinsic_target_type,
            extrinsic_target_type=routing.extrinsic_target_type,
            extrinsic_charuco_same_as_intrinsic=same,
        )
        self.save_routing(new_routing)

    # -- Chessboard -------------------------------------------------------

    def load_chessboard(self) -> Chessboard:
        """Load chessboard config. Raises ValueError if file doesn't exist."""
        path = self._dir / "chessboard.toml"
        try:
            return Chessboard.from_toml(path)
        except PersistenceError as e:
            raise ValueError(f"Failed to load chessboard: {e}") from e

    def save_chessboard(self, chessboard: Chessboard) -> None:
        """Save chessboard config to chessboard.toml."""
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            chessboard.to_toml(self._dir / "chessboard.toml")
        except PersistenceError as e:
            raise ValueError(f"Failed to save chessboard: {e}") from e

    def chessboard_exists(self) -> bool:
        """Check if chessboard.toml exists."""
        return (self._dir / "chessboard.toml").exists()

    # -- ArUco Target -----------------------------------------------------

    def load_aruco_target(self) -> ArucoTarget:
        """Load ArUco target config. Raises ValueError if file doesn't exist."""
        path = self._dir / "aruco_target.toml"
        try:
            return ArucoTarget.from_toml(path)
        except PersistenceError as e:
            raise ValueError(f"Failed to load aruco target: {e}") from e

    def save_aruco_target(self, target: ArucoTarget) -> None:
        """Save ArUco target config to aruco_target.toml."""
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            target.to_toml(self._dir / "aruco_target.toml")
        except PersistenceError as e:
            raise ValueError(f"Failed to save aruco target: {e}") from e

    def aruco_target_exists(self) -> bool:
        """Check if aruco_target.toml exists."""
        return (self._dir / "aruco_target.toml").exists()

    # -- Convenience: Role-Based Tracker Name -----------------------------

    def get_extrinsic_tracker_name(self) -> str:
        """Return the tracker name string for the current extrinsic target type.

        Returns "CHARUCO" or "ARUCO". Used for:
        - The extraction output subfolder name (extrinsic/CHARUCO/ or extrinsic/ARUCO/)
        - The extrinsic_image_points_path computation
        """
        routing = self.get_routing()
        if routing.extrinsic_target_type == "charuco":
            return "CHARUCO"
        else:
            return "ARUCO"

    # -- Initialization ---------------------------------------------------

    def initialize_defaults(self) -> None:
        """Create default config and target files if they don't exist.

        Called once during WorkspaceCoordinator._initialize_project_files().

        Defaults:
        - config.toml: charuco/charuco, same_as_intrinsic = true
        - intrinsic_charuco.toml: Charuco(4, 5, 11, 8.5, square_size_override_cm=5.4)
        - chessboard.toml: Chessboard(rows=6, columns=9)
        - aruco_target.toml: ArucoTarget.single_marker()

        Note: extrinsic_charuco.toml is NOT created at init because
        same_as_intrinsic defaults to true (reads from intrinsic file).
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        # Routing config (always create/update to ensure defaults)
        if not self._config_path.exists():
            logger.info("Creating default calibration target routing config")
            default_routing = TargetRouting(
                intrinsic_target_type="charuco",
                extrinsic_target_type="charuco",
                extrinsic_charuco_same_as_intrinsic=True,
            )
            self.save_routing(default_routing)

        # Intrinsic charuco
        if not self.intrinsic_charuco_exists():
            logger.info("Creating default intrinsic charuco board")
            default_charuco = Charuco(4, 5, 11, 8.5, square_size_override_cm=5.4)
            self.save_intrinsic_charuco(default_charuco)

        # Chessboard
        if not self.chessboard_exists():
            logger.info("Creating default chessboard pattern")
            default_chessboard = Chessboard(rows=6, columns=9)
            self.save_chessboard(default_chessboard)

        # ArUco target
        if not self.aruco_target_exists():
            logger.info("Creating default ArUco target")
            import cv2

            default_aruco = ArucoTarget.single_marker(
                marker_id=0,
                marker_size_m=0.05,
                dictionary=cv2.aruco.DICT_4X4_100,
            )
            self.save_aruco_target(default_aruco)
