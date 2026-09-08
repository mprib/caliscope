"""Tests for the vertex layouts of custom Qt3D primitives."""

import numpy as np
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DRender import Qt3DRender
from PySide6.QtGui import QColor

from caliscope.gui.qt3d.primitives import create_double_sided_mesh, create_line_entity


def _geometry_attributes(entity: Qt3DCore.QEntity) -> dict[str, Qt3DCore.QAttribute]:
    renderer = next(
        component
        for component in entity.components()
        if isinstance(component, Qt3DRender.QGeometryRenderer)
    )
    geometry = renderer.geometry()
    assert geometry is not None
    return {attribute.name(): attribute for attribute in geometry.attributes()}


def _assert_normal_attribute(entity: Qt3DCore.QEntity, vertex_count: int) -> None:
    normal = _geometry_attributes(entity)[Qt3DCore.QAttribute.defaultNormalAttributeName()]

    assert normal.vertexBaseType() == Qt3DCore.QAttribute.VertexBaseType.Float
    assert normal.vertexSize() == 3
    assert normal.byteStride() == 3 * np.dtype(np.float32).itemsize
    assert normal.count() == vertex_count
    np.testing.assert_array_equal(
        np.frombuffer(normal.buffer().data().data(), dtype=np.float32).reshape(vertex_count, 3),
        np.tile((0.0, 0.0, 1.0), (vertex_count, 1)),
    )


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

    _assert_normal_attribute(line, vertex_count=2)
    _assert_normal_attribute(mesh, vertex_count=3)
