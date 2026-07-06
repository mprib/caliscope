"""Tests for WorkspaceCoordinator's extrinsic constraint-factory wiring.

Charuco calibration used to run with constraints=None: board geometry seeded
PnP but never entered bundle adjustment as a constraint. These tests confirm
create_extrinsic_calibration_presenter() wires a constraint factory that
compiles board-geometry distance constraints for the default (charuco)
target, and still wires the ArUco marker-set factory for the ArUco target.
"""

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from caliscope.workspace_coordinator import WorkspaceCoordinator


@pytest.fixture
def qapp():
    """Ensure QCoreApplication exists for Qt signal/QObject construction."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


@pytest.fixture
def coordinator(tmp_path: Path, qapp) -> WorkspaceCoordinator:
    return WorkspaceCoordinator(tmp_path)


def test_charuco_extrinsic_presenter_gets_board_geometry_constraints(coordinator: WorkspaceCoordinator):
    """Default routing is charuco/charuco; the presenter's constraint factory
    should compile a non-empty ConstraintSet from board geometry, not None."""
    assert coordinator.targets_repository.extrinsic_target_type == "charuco"

    presenter = coordinator.create_extrinsic_calibration_presenter()

    assert presenter._constraint_factory is not None
    constraints = presenter._constraint_factory()
    assert constraints is not None
    assert len(constraints.distances) > 0
    assert constraints.static_object_ids == frozenset()
    assert constraints.centroid_distances == ()
    assert presenter._extrinsic_target_type == "charuco"


def test_aruco_extrinsic_presenter_gets_marker_set_constraints(coordinator: WorkspaceCoordinator):
    """Switching routing to aruco keeps the marker-set constraint factory
    (regression check for the branch this task modified)."""
    routing = coordinator.targets_repository.get_routing()
    coordinator.targets_repository.save_routing(
        type(routing)(
            intrinsic_target_type=routing.intrinsic_target_type,
            extrinsic_target_type="aruco",
            extrinsic_charuco_same_as_intrinsic=routing.extrinsic_charuco_same_as_intrinsic,
        )
    )

    presenter = coordinator.create_extrinsic_calibration_presenter()

    assert presenter._constraint_factory is not None
    constraints = presenter._constraint_factory()
    assert constraints is not None
    assert len(constraints.distances) > 0
    assert presenter._extrinsic_target_type == "aruco"
