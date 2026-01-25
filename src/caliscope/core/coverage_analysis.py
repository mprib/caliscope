"""Coverage analysis functions for calibration quality assessment.

Provides pure functions for analyzing coverage data before calibration,
helping users understand data quality and identify issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from caliscope.core.point_data import ImagePoints


class LinkQuality(Enum):
    """Quality classification for camera pair linkage."""

    GOOD = "good"  # >1000 obs, >50 frames, density >0.6
    ADEQUATE = "adequate"  # 500-1000 obs, 30-50 frames, density 0.4-0.6
    WEAK = "weak"  # 100-500 obs, 10-30 frames, density 0.1-0.4
    CRITICAL = "critical"  # <100 obs or <10 frames or density <0.1
    DISCONNECTED = "disconnected"  # 0 shared observations


class TopologyClass(Enum):
    """Classification of camera network topology."""

    RING = "ring"  # Every camera sees 2+ neighbors, no articulation points
    CHAIN = "chain"  # Linear, each camera only sees neighbors
    STAR = "star"  # One central camera sees all, others only see center
    FRAGMENTED = "fragmented"  # Multiple disconnected components


@dataclass(frozen=True)
class CoverageThresholds:
    """Configurable thresholds for coverage quality assessment."""

    observations_critical: int = 100
    observations_weak: int = 500
    observations_adequate: int = 1000
    frames_critical: int = 10
    frames_weak: int = 30
    frames_adequate: int = 50
    density_critical: float = 0.1
    density_weak: float = 0.4
    density_adequate: float = 0.6


@dataclass(frozen=True)
class ExtrinsicCoverageReport:
    """Multi-camera coverage analysis for extrinsic calibration.

    Assesses whether camera pairs have sufficient shared observations
    for stereo calibration.
    """

    # Raw data matrices (n_cameras x n_cameras, symmetric)
    pairwise_observations: NDArray[np.int64]  # Shared observation counts
    pairwise_frames: NDArray[np.int64]  # Shared frame counts
    observation_density: NDArray[np.float64]  # obs / (frames * max_corners)

    # Graph topology analysis
    isolated_cameras: list[int]  # Ports with zero shared observations
    weak_links: list[tuple[int, int, LinkQuality]]  # Camera pairs below "adequate"
    articulation_points: list[int]  # Cameras whose removal disconnects graph
    bridge_edges: list[tuple[int, int]]  # Links that are only path between cameras

    # Summary metrics
    redundancy_factor: float  # total_edges / (n_cameras - 1), 1.0 = minimal tree
    topology_class: TopologyClass

    @property
    def n_cameras(self) -> int:
        """Number of cameras in the analysis."""
        return len(self.pairwise_observations)

    @property
    def has_critical_issues(self) -> bool:
        """True if there are isolated cameras or critical weak links."""
        if self.isolated_cameras:
            return True
        return any(quality == LinkQuality.CRITICAL for _, _, quality in self.weak_links)


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


def analyze_multi_camera_coverage(
    image_points: ImagePoints,
    max_corners_per_frame: int = 35,  # Typical charuco board
    thresholds: CoverageThresholds | None = None,
) -> ExtrinsicCoverageReport:
    """Analyze pairwise coverage for extrinsic calibration.

    Discovers actual camera ports from the data and builds port-to-index mapping
    internally. Reports (isolated cameras, weak links, etc.) use actual port numbers.

    Args:
        image_points: ImagePoints containing all camera observations
        max_corners_per_frame: Expected max corners per frame (for density calc)
        thresholds: Quality thresholds (uses defaults if None)

    Returns:
        ExtrinsicCoverageReport with comprehensive coverage analysis
    """
    if thresholds is None:
        thresholds = CoverageThresholds()

    df = image_points.df

    # Build port-to-index mapping from actual data
    # This handles arbitrary port numbers (not just 0..n-1)
    actual_ports = sorted(df["port"].unique()) if len(df) > 0 else []
    port_to_index = {port: idx for idx, port in enumerate(actual_ports)}
    index_to_port = {idx: port for port, idx in port_to_index.items()}
    n_actual = len(actual_ports)

    # Compute observation counts using actual port mapping
    pairwise_obs = compute_coverage_matrix(image_points, port_to_index)

    # Compute frame counts
    pairwise_frames = np.zeros((n_actual, n_actual), dtype=np.int64)
    if len(df) > 0:
        frame_groups = df.groupby("sync_index")["port"].apply(set)
        for ports in frame_groups:
            port_list = sorted(ports)
            for i, port_i in enumerate(port_list):
                for port_j in port_list[i:]:
                    if port_i in port_to_index and port_j in port_to_index:
                        idx_i = port_to_index[port_i]
                        idx_j = port_to_index[port_j]
                        pairwise_frames[idx_i, idx_j] += 1
                        if idx_i != idx_j:
                            pairwise_frames[idx_j, idx_i] += 1

    # Compute observation density
    with np.errstate(divide="ignore", invalid="ignore"):
        density = pairwise_obs / (pairwise_frames * max_corners_per_frame)
        density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)

    # Find isolated cameras and classify links
    # Report using actual port numbers, not matrix indices
    isolated: list[int] = []
    weak_links: list[tuple[int, int, LinkQuality]] = []

    for idx_i in range(n_actual):
        has_any_link = False
        for idx_j in range(n_actual):
            if idx_i == idx_j:
                continue
            obs = pairwise_obs[idx_i, idx_j]
            frames = pairwise_frames[idx_i, idx_j]
            dens = density[idx_i, idx_j]

            if obs > 0:
                has_any_link = True

            # Only record each pair once (idx_i < idx_j)
            if idx_i < idx_j:
                quality = _classify_link_quality(obs, frames, dens, thresholds)
                if quality in (LinkQuality.WEAK, LinkQuality.CRITICAL, LinkQuality.DISCONNECTED):
                    # Convert back to actual port numbers
                    port_i = index_to_port[idx_i]
                    port_j = index_to_port[idx_j]
                    weak_links.append((port_i, port_j, quality))

        if not has_any_link:
            # Convert back to actual port number
            isolated.append(index_to_port[idx_i])

    # Graph analysis using NetworkX
    articulation_pts_idx, bridges_idx = _compute_graph_metrics(pairwise_obs)
    # Convert back to actual port numbers
    articulation_pts = [index_to_port[idx] for idx in articulation_pts_idx]
    bridges = [(index_to_port[i], index_to_port[j]) for i, j in bridges_idx]

    # Compute redundancy and topology
    n_edges = np.sum(pairwise_obs > 0) // 2  # Symmetric matrix, count once
    min_edges = max(n_actual - 1, 1)
    redundancy = n_edges / min_edges if min_edges > 0 else 0.0

    topology = _classify_topology(pairwise_obs, articulation_pts_idx, isolated)

    return ExtrinsicCoverageReport(
        pairwise_observations=pairwise_obs,
        pairwise_frames=pairwise_frames,
        observation_density=density,
        isolated_cameras=isolated,
        weak_links=weak_links,
        articulation_points=articulation_pts,
        bridge_edges=bridges,
        redundancy_factor=float(redundancy),
        topology_class=topology,
    )


def _classify_link_quality(
    obs: int,
    frames: int,
    density: float,
    thresholds: CoverageThresholds,
) -> LinkQuality:
    """Classify quality of a camera pair link."""
    if obs == 0:
        return LinkQuality.DISCONNECTED

    if (
        obs < thresholds.observations_critical
        or frames < thresholds.frames_critical
        or density < thresholds.density_critical
    ):
        return LinkQuality.CRITICAL

    if obs < thresholds.observations_weak or frames < thresholds.frames_weak or density < thresholds.density_weak:
        return LinkQuality.WEAK

    if (
        obs < thresholds.observations_adequate
        or frames < thresholds.frames_adequate
        or density < thresholds.density_adequate
    ):
        return LinkQuality.ADEQUATE

    return LinkQuality.GOOD


def _compute_graph_metrics(
    adjacency: NDArray[np.int64],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Compute articulation points and bridges using NetworkX."""
    G = nx.Graph()
    n = len(adjacency)

    # Add all nodes first
    G.add_nodes_from(range(n))

    # Add edges where observations exist
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                G.add_edge(i, j)

    articulation_points = list(nx.articulation_points(G)) if G.number_of_edges() > 0 else []
    bridges = list(nx.bridges(G)) if G.number_of_edges() > 0 else []

    return articulation_points, bridges


def _classify_topology(
    adjacency: NDArray[np.int64],
    articulation_points: list[int],
    isolated: list[int],
) -> TopologyClass:
    """Classify the camera network topology."""
    n = len(adjacency)

    if isolated:
        return TopologyClass.FRAGMENTED

    # Check connectivity
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                G.add_edge(i, j)

    if not nx.is_connected(G):
        return TopologyClass.FRAGMENTED

    # Check for star topology (one node with n-1 edges, others with 1)
    # NetworkX stubs incorrectly type G.degree() as int; it's actually DegreeView
    degree_dict = dict(G.degree())  # type: ignore[call-overload]
    degrees = [degree_dict[i] for i in range(n)]
    if max(degrees) == n - 1 and degrees.count(1) == n - 1:
        return TopologyClass.STAR

    # Check for chain (all articulation points, each node has <= 2 edges)
    if len(articulation_points) == n - 2 and all(d <= 2 for d in degrees):
        return TopologyClass.CHAIN

    # Ring if no articulation points and each camera sees 2+ neighbors
    if not articulation_points and all(d >= 2 for d in degrees):
        return TopologyClass.RING

    # Default to ring if well-connected
    return TopologyClass.RING


def generate_multi_camera_guidance(report: ExtrinsicCoverageReport) -> list[str]:
    """Return actionable guidance for improving coverage."""
    messages: list[str] = []

    if report.isolated_cameras:
        ports = ", ".join(f"C{p}" for p in report.isolated_cameras)
        messages.append(f"CRITICAL: Cameras {ports} have no shared observations with any other camera")

    critical_links = [(a, b) for a, b, q in report.weak_links if q == LinkQuality.CRITICAL]
    if critical_links:
        pairs = ", ".join(f"C{a}-C{b}" for a, b in critical_links[:3])
        messages.append(f"Camera pairs {pairs} have critically low shared observations")

    if report.articulation_points and report.topology_class != TopologyClass.CHAIN:
        ports = ", ".join(f"C{p}" for p in report.articulation_points[:2])
        messages.append(f"Network depends on cameras {ports} - capture more overlapping views for redundancy")

    return messages
