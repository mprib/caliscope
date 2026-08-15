"""Tests for the vertex layouts of custom Qt3D primitives."""

import numpy as np
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
from PySide6.QtGui import QColor

from caliscope.gui.qt3d.primitives import create_double_sided_mesh, create_line_entity


def _geometry_attribute_names(entity: Qt3DCore.QEntity) -> set[str]:
    renderer = next(
        component
        for component in entity.components()
        if isinstance(component, Qt3DRender.QGeometryRenderer)
    )
    geometry = renderer.geometry()
    assert geometry is not None
    return {attribute.name() for attribute in geometry.attributes()}


def test_custom_primitives_provide_phong_shader_attributes(qapp) -> None:  # noqa: ARG001
    """Custom geometry supplies every vertex attribute required by QPhongMaterial."""
    root = Qt3DCore.QEntity()
    vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32)

    line, _ = create_line_entity(
        vertices[:2],
        np.array((0, 1), dtype=np.uint32),
        QColor("red"),
        root,
    )
    mesh = create_double_sided_mesh(
        vertices,
        np.array((0, 1, 2), dtype=np.uint32),
        QColor("green"),
        root,
    )

    required_attributes = {
        Qt3DCore.QAttribute.defaultPositionAttributeName(),
        Qt3DCore.QAttribute.defaultNormalAttributeName(),
    }
    assert required_attributes <= _geometry_attribute_names(line)
    assert required_attributes <= _geometry_attribute_names(mesh)
