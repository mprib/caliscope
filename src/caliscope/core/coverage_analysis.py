"""Coverage analysis functions for calibration quality assessment.

Provides pure functions for analyzing coverage data before calibration,
helping users understand data quality and identify issues.

Design Philosophy:
- The heatmap IS the feedback - observation counts tell the full story
- Only warn about truly actionable structural issues (disconnected cameras, islands)
- Don't classify topology or redundancy - not actionable and often misleading
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from caliscope.core.point_data import ImagePoints


class LinkQuality(Enum):
    """Quality classification for camera pair linkage.

    Simplified to 3 levels focused on actionability.
    """

    GOOD = "good"  # >= 200 shared observations
    MARGINAL = "marginal"  # 50-200 shared observations
    INSUFFICIENT = "insufficient"  # < 50 shared observations


# Thresholds for link quality classification
GOOD_OBSERVATION_THRESHOLD = 200
MARGINAL_OBSERVATION_THRESHOLD = 50


class WarningSeverity(Enum):
    """Severity levels for structural warnings."""

    CRITICAL = "critical"  # Calibration will fail
    WARNING = "warning"  # May cause issues
    INFO = "info"  # Informational


@dataclass(frozen=True)
class StructuralWarning:
    """A structural issue in the camera network."""

    severity: WarningSeverity
    message: str


@dataclass(frozen=True)
class ExtrinsicCoverageReport:
    """Multi-camera coverage analysis for extrinsic calibration.

    Focused on data the UI actually needs:
    - pairwise_observations: For heatmap visualization
    - isolated_cameras: Critical failure condition
    - n_connected_components: Should be 1 for successful calibration
    - leaf_cameras: Cameras with only one connection (potentially fragile)

    Removed over-engineered metrics (topology class, articulation points,
    bridges, redundancy factor) - not actionable and often misleading.
    """

    # Raw data matrix (n_cameras x n_cameras, symmetric)
    pairwise_observations: NDArray[np.int64]  # Shared observation counts

    # Structural analysis (using actual port numbers)
    isolated_cameras: list[int]  # Ports with zero shared observations
    n_connected_components: int  # Should be 1
    leaf_cameras: list[tuple[int, int, int]]  # (port, connected_to, obs_count)

    @property
    def n_cameras(self) -> int:
        """Number of cameras in the analysis."""
        return len(self.pairwise_observations)

    @property
    def has_critical_issues(self) -> bool:
        """True if calibration will definitely fail."""
        return bool(self.isolated_cameras) or self.n_connected_components > 1


def compute_coverage_matrix(
    image_points: ImagePoints,
    port_to_index: dict[int, int],
) -> NDArray[np.int64]:
    """Compute camera-pair shared observation counts.

    The coverage matrix is an (n_cameras, n_cameras) symmetric matrix where:
    - Diagonal [i, i]: Total observations from camera i
    - Off-diagonal [i, j]: Count of (sync_index, point_id) pairs seen by BOTH cameras

    Args:
        image_points: ImagePoints to analyze
        port_to_index: Mapping from actual port numbers to matrix indices

    Returns:
        (n_cameras, n_cameras) symmetric matrix of observation counts
    """
    df = image_points.df
    n_cameras = len(port_to_index)
    coverage = np.zeros((n_cameras, n_cameras), dtype=np.int64)

    # Group by (sync_index, point_id) to find which cameras see each point
    grouped = df.groupby(["sync_index", "point_id"])["port"].apply(set)

    for ports in grouped:
        port_list = sorted(ports)
        for i, port_i in enumerate(port_list):
            for port_j in port_list[i:]:
                if port_i in port_to_index and port_j in port_to_index:
                    idx_i = port_to_index[port_i]
                    idx_j = port_to_index[port_j]
                    coverage[idx_i, idx_j] += 1
                    if idx_i != idx_j:
                        coverage[idx_j, idx_i] += 1

    return coverage


def _find_connected_components(adjacency: NDArray[np.int64]) -> list[set[int]]:
    """Find connected components using BFS (no NetworkX needed).

    Args:
        adjacency: (n, n) adjacency matrix where >0 means connected

    Returns:
        List of sets, each set containing indices in a connected component
    """
    n = len(adjacency)
    visited = [False] * n
    components: list[set[int]] = []

    for start in range(n):
        if visited[start]:
            continue

        # BFS from this node
        component: set[int] = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            component.add(node)

            # Add unvisited neighbors
            for neighbor in range(n):
                if not visited[neighbor] and adjacency[node, neighbor] > 0:
                    queue.append(neighbor)

        components.append(component)

    return components


def _find_leaf_cameras(
    adjacency: NDArray[np.int64],
    index_to_port: dict[int, int],
) -> list[tuple[int, int, int]]:
    """Find cameras with exactly one connection (leaf nodes).

    Args:
        adjacency: (n, n) adjacency matrix
        index_to_port: Mapping from matrix indices to actual port numbers

    Returns:
        List of (port, connected_to_port, observation_count) tuples
    """
    n = len(adjacency)
    leaf_cameras: list[tuple[int, int, int]] = []

    for idx in range(n):
        # Count neighbors (non-zero off-diagonal entries)
        neighbors = [(j, adjacency[idx, j]) for j in range(n) if j != idx and adjacency[idx, j] > 0]

        if len(neighbors) == 1:
            connected_idx, obs_count = neighbors[0]
            port = index_to_port[idx]
            connected_port = index_to_port[connected_idx]
            leaf_cameras.append((port, connected_port, int(obs_count)))

    return leaf_cameras


def analyze_multi_camera_coverage(
    image_points: ImagePoints,
) -> ExtrinsicCoverageReport:
    """Analyze pairwise coverage for extrinsic calibration.

    Discovers actual camera ports from the data and builds port-to-index mapping
    internally. Reports use actual port numbers, not matrix indices.

    Args:
        image_points: ImagePoints containing all camera observations

    Returns:
        ExtrinsicCoverageReport with coverage analysis
    """
    df = image_points.df

    # Build port-to-index mapping from actual data
    actual_ports = sorted(df["port"].unique()) if len(df) > 0 else []
    port_to_index = {port: idx for idx, port in enumerate(actual_ports)}
    index_to_port = {idx: port for port, idx in port_to_index.items()}

    # Compute observation counts using actual port mapping
    pairwise_obs = compute_coverage_matrix(image_points, port_to_index)

    # Find isolated cameras (no connections at all)
    isolated: list[int] = []
    for idx in range(len(actual_ports)):
        has_any_link = any(pairwise_obs[idx, j] > 0 for j in range(len(actual_ports)) if j != idx)
        if not has_any_link:
            isolated.append(index_to_port[idx])

    # Find connected components
    components = _find_connected_components(pairwise_obs)

    # Find leaf cameras
    leaf_cameras = _find_leaf_cameras(pairwise_obs, index_to_port)

    return ExtrinsicCoverageReport(
        pairwise_observations=pairwise_obs,
        isolated_cameras=isolated,
        n_connected_components=len(components),
        leaf_cameras=leaf_cameras,
    )


def classify_link_quality(observation_count: int) -> LinkQuality:
    """Classify the quality of a camera pair link based on observation count."""
    if observation_count >= GOOD_OBSERVATION_THRESHOLD:
        return LinkQuality.GOOD
    elif observation_count >= MARGINAL_OBSERVATION_THRESHOLD:
        return LinkQuality.MARGINAL
    else:
        return LinkQuality.INSUFFICIENT


def detect_structural_warnings(
    report: ExtrinsicCoverageReport,
    n_cameras: int,
    min_leaf_observations: int = 100,
) -> list[StructuralWarning]:
    """Detect actionable structural issues in camera network.

    Args:
        report: Coverage analysis report
        n_cameras: Number of cameras in the setup
        min_leaf_observations: Threshold for warning about weak leaf connections

    Returns:
        List of warnings sorted by severity (critical first)
    """
    warnings: list[StructuralWarning] = []

    # Critical: Disconnected cameras
    for port in report.isolated_cameras:
        warnings.append(
            StructuralWarning(
                WarningSeverity.CRITICAL,
                f"Camera C{port} has no shared observations with any other camera",
            )
        )

    # Critical: Multiple islands
    if report.n_connected_components > 1:
        warnings.append(
            StructuralWarning(
                WarningSeverity.CRITICAL,
                f"Camera network has {report.n_connected_components} disconnected groups",
            )
        )

    # Leaf node warnings (skip for 2-camera setup - both are necessarily leaves)
    if n_cameras > 2:
        for port, connected_to, obs_count in report.leaf_cameras:
            if obs_count < min_leaf_observations:
                warnings.append(
                    StructuralWarning(
                        WarningSeverity.WARNING,
                        f"Camera C{port} only connected to C{connected_to} ({obs_count} obs)",
                    )
                )
            else:
                warnings.append(
                    StructuralWarning(
                        WarningSeverity.INFO,
                        f"Camera C{port} connects only through C{connected_to}",
                    )
                )

    # Sort by severity (critical first)
    severity_order = {
        WarningSeverity.CRITICAL: 0,
        WarningSeverity.WARNING: 1,
        WarningSeverity.INFO: 2,
    }
    warnings.sort(key=lambda w: severity_order[w.severity])

    return warnings
