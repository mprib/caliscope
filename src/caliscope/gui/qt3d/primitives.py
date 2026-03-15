"""Low-level Qt3D entity builders for 3D visualization.

Converts numpy arrays into Qt3D entities (meshes, lines, sphere clouds, grids).
No domain knowledge — purely geometric primitives.
"""

import numpy as np
from PySide6.QtCore import QByteArray
from PySide6.QtGui import QColor, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.Qt3DRender import Qt3DRender


def numpy_to_qbytearray(arr: np.ndarray) -> QByteArray:
    """Convert numpy array to QByteArray for Qt3D buffers."""
    return QByteArray(arr.astype(np.float32).tobytes())


def create_line_entity(
    vertices: np.ndarray,
    indices: np.ndarray,
    color: QColor,
    parent: Qt3DCore.QEntity,
) -> tuple[Qt3DCore.QEntity, Qt3DCore.QBuffer]:
    """Create a line-drawing entity from vertex positions and index pairs.

    Returns (entity, vertex_buffer) — retain the buffer handle to update
    vertex positions per-frame via buffer.setData().

    Args:
        vertices: (N, 3) float32 vertex positions.
        indices: flat uint32 index array [a0, b0, a1, b1, ...] for Lines primitive.
        color: Line color.
        parent: Parent entity in the scene graph.
    """
    entity = Qt3DCore.QEntity(parent)
    geometry = Qt3DCore.QGeometry(entity)

    vertex_buf = Qt3DCore.QBuffer(geometry)
    vertex_buf.setData(numpy_to_qbytearray(vertices))

    pos_attr = Qt3DCore.QAttribute(geometry)
    pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
    pos_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
    pos_attr.setVertexSize(3)
    pos_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.VertexAttribute)
    pos_attr.setBuffer(vertex_buf)
    pos_attr.setByteStride(3 * 4)
    pos_attr.setCount(len(vertices))
    geometry.addAttribute(pos_attr)

    index_data = indices.astype(np.uint32)
    index_buf = Qt3DCore.QBuffer(geometry)
    index_buf.setData(QByteArray(index_data.tobytes()))

    index_attr = Qt3DCore.QAttribute(geometry)
    index_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.UnsignedInt)
    index_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.IndexAttribute)
    index_attr.setBuffer(index_buf)
    index_attr.setCount(len(index_data))
    geometry.addAttribute(index_attr)

    renderer = Qt3DRender.QGeometryRenderer(entity)
    renderer.setGeometry(geometry)
    renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.PrimitiveType.Lines)

    material = Qt3DExtras.QPhongMaterial(entity)
    material.setAmbient(color)
    material.setDiffuse(color)

    entity.addComponent(renderer)
    entity.addComponent(material)

    return entity, vertex_buf


def create_double_sided_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
    color: QColor,
    parent: Qt3DCore.QEntity,
    opacity: float = 0.6,
) -> Qt3DCore.QEntity:
    """Create a triangle mesh visible from both sides.

    Duplicates triangles with reversed winding order for double-sided rendering.
    Uses QPhongAlphaMaterial for transparency support.
    """
    reversed_indices = indices.reshape(-1, 3)[:, ::-1].flatten()
    all_indices = np.concatenate([indices, reversed_indices]).astype(np.uint32)

    entity = Qt3DCore.QEntity(parent)
    geometry = Qt3DCore.QGeometry(entity)

    vertex_buf = Qt3DCore.QBuffer(geometry)
    vertex_buf.setData(numpy_to_qbytearray(vertices))

    pos_attr = Qt3DCore.QAttribute(geometry)
    pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
    pos_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
    pos_attr.setVertexSize(3)
    pos_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.VertexAttribute)
    pos_attr.setBuffer(vertex_buf)
    pos_attr.setByteStride(3 * 4)
    pos_attr.setCount(len(vertices))
    geometry.addAttribute(pos_attr)

    index_buf = Qt3DCore.QBuffer(geometry)
    index_buf.setData(QByteArray(all_indices.tobytes()))

    index_attr = Qt3DCore.QAttribute(geometry)
    index_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.UnsignedInt)
    index_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.IndexAttribute)
    index_attr.setBuffer(index_buf)
    index_attr.setCount(len(all_indices))
    geometry.addAttribute(index_attr)

    renderer = Qt3DRender.QGeometryRenderer(entity)
    renderer.setGeometry(geometry)
    renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.PrimitiveType.Triangles)

    material = Qt3DExtras.QPhongAlphaMaterial(entity)
    material.setAmbient(color)
    material.setDiffuse(color)
    material.setAlpha(opacity)

    entity.addComponent(renderer)
    entity.addComponent(material)

    return entity


class SphereCloud:
    """Collection of sphere entities that can be repositioned per-frame.

    Each point gets its own QEntity with a QSphereMesh and QTransform.
    For motion capture landmark counts (10-50 points), individual entities
    are simpler and more reliable than instanced rendering.

    NaN positions move the sphere off-screen (Qt3D lacks a cheap per-entity
    visibility toggle without removing components).
    """

    _HIDDEN_POS = QVector3D(99999, 99999, 99999)

    def __init__(
        self,
        n_points: int,
        color: QColor,
        parent: Qt3DCore.QEntity,
        sphere_radius: float = 0.02,
    ):
        self._parent_entity = Qt3DCore.QEntity(parent)
        self._transforms: list[Qt3DCore.QTransform] = []

        for _ in range(n_points):
            entity = Qt3DCore.QEntity(self._parent_entity)

            mesh = Qt3DExtras.QSphereMesh(entity)
            mesh.setRadius(sphere_radius)
            mesh.setRings(8)
            mesh.setSlices(8)

            material = Qt3DExtras.QPhongMaterial(entity)
            material.setAmbient(color)
            material.setDiffuse(color)

            transform = Qt3DCore.QTransform(entity)
            transform.setTranslation(self._HIDDEN_POS)

            entity.addComponent(mesh)
            entity.addComponent(material)
            entity.addComponent(transform)

            self._transforms.append(transform)

    def update_positions(self, positions: np.ndarray) -> None:
        """Update sphere positions. NaN positions hide the sphere off-screen."""
        for i, transform in enumerate(self._transforms):
            pos = positions[i]
            if np.isnan(pos[0]):
                transform.setTranslation(self._HIDDEN_POS)
            else:
                transform.setTranslation(QVector3D(float(pos[0]), float(pos[1]), float(pos[2])))

    def set_enabled(self, enabled: bool) -> None:
        """Show or hide the entire point cloud."""
        self._parent_entity.setEnabled(enabled)


def build_floor_grid(
    parent: Qt3DCore.QEntity,
    size: float = 5.0,
    spacing: float = 0.5,
    color: QColor = QColor(80, 80, 80),
    axis_color: QColor = QColor(120, 120, 120),
) -> None:
    """Add a grid at z=0 for spatial reference.

    Lines through the origin are drawn brighter to mark the principal axes.
    """
    half = size / 2
    n_lines = int(size / spacing)

    idx = 0
    regular_verts: list[list[float]] = []
    regular_indices: list[int] = []

    for i in range(n_lines + 1):
        coord = -half + i * spacing
        if abs(coord) < spacing * 0.01:
            continue  # Skip origin lines — drawn separately as brighter

        regular_verts.extend(
            [
                [-half, coord, 0],
                [half, coord, 0],
            ]
        )
        regular_indices.extend([idx, idx + 1])
        idx += 2

        regular_verts.extend(
            [
                [coord, -half, 0],
                [coord, half, 0],
            ]
        )
        regular_indices.extend([idx, idx + 1])
        idx += 2

    if regular_verts:
        verts_arr = np.array(regular_verts, dtype=np.float32)
        idx_arr = np.array(regular_indices, dtype=np.uint32)
        create_line_entity(verts_arr, idx_arr, color, parent)

    # Origin axis lines (brighter)
    axis_verts = np.array(
        [
            [-half, 0, 0],
            [half, 0, 0],
            [0, -half, 0],
            [0, half, 0],
        ],
        dtype=np.float32,
    )
    axis_indices = np.array([0, 1, 2, 3], dtype=np.uint32)
    create_line_entity(axis_verts, axis_indices, axis_color, parent)


def build_origin_axes(parent: Qt3DCore.QEntity, length: float = 0.3) -> None:
    """Add XYZ axis indicators at the origin using colored cylinders."""
    axes = [
        (QVector3D(1, 0, 0), QColor("red"), QVector3D(0, 0, -90)),  # X
        (QVector3D(0, 1, 0), QColor("green"), QVector3D(0, 0, 0)),  # Y
        (QVector3D(0, 0, 1), QColor("blue"), QVector3D(90, 0, 0)),  # Z
    ]

    for direction, color, euler_rot in axes:
        entity = Qt3DCore.QEntity(parent)

        mesh = Qt3DExtras.QCylinderMesh(entity)
        mesh.setRadius(0.008)
        mesh.setLength(length)

        material = Qt3DExtras.QPhongMaterial(entity)
        material.setAmbient(color)
        material.setDiffuse(color)

        transform = Qt3DCore.QTransform(entity)
        offset = direction * (length / 2)
        transform.setTranslation(offset)
        transform.setRotationX(euler_rot.x())
        transform.setRotationZ(euler_rot.z())

        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)
