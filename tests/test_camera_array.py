""" """

import logging
from pathlib import Path

import numpy as np

from caliscope import __root__
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.core.point_data import ImagePoints
from caliscope.cameras.camera_array import CameraArray

logger = logging.getLogger(__name__)


def analyze_true_connectivity(image_points: ImagePoints) -> dict:
    """
    Analyze the raw data to determine which camera pairs SHOULD be linked
    based on actual shared observations.
    """
    df = image_points.df

    print("\n" + "=" * 80)
    print("TRUE CONNECTIVITY ANALYSIS")
    print("=" * 80)

    # 1. Points per camera
    print("\n📊 OBSERVATIONS PER CAMERA:")
    obs_per_cam = df.groupby("cam_id").size().sort_index()
    for cam_id, count in obs_per_cam.items():
        print(f"  Camera {cam_id}: {count:,} observations")

    # 2. Shared point analysis
    print("\n🔍 SHARED POINTS ANALYSIS:")

    # Group by sync_index + (object_id, keypoint_id) to find which cameras see the same point
    point_coverage = df.groupby(["sync_index", "object_id", "keypoint_id"])["cam_id"].apply(
        lambda x: tuple(sorted(set(x)))
    )

    # Count coverage patterns
    coverage_counts = point_coverage.value_counts()

    print("\n  Coverage patterns (which cameras see same point):")
    for pattern, count in coverage_counts.head(20).items():
        print(f"    {pattern}: {count} points")

    # 3. Direct pair connectivity
    print("\n🔗 DIRECT PAIR CONNECTIVITY:")

    # For each sync_index, find all camera pairs that see the same board
    pair_counts = {}
    for (sync_idx, object_id, keypoint_id), group in df.groupby(["sync_index", "object_id", "keypoint_id"]):
        ports_in_group = sorted(group["cam_id"].unique())

        # Count all pairs within this group
        for i, port_a in enumerate(ports_in_group):
            for port_b in ports_in_group[i + 1 :]:
                pair = tuple(sorted((port_a, port_b)))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Sort by count descending
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    print("\n  Port pairs that share points (sorted by shared point count):")
    for (port_a, port_b), count in sorted_pairs:
        print(f"    {port_a}-{port_b}: {count:,} shared points")

    # 4. Board-level connectivity
    print("\n🎯 BOARD-LEVEL CONNECTIVITY:")

    # For each camera pair, count how many unique boards (sync_index) they share
    board_counts = {}
    for (sync_idx, object_id, keypoint_id), group in df.groupby(["sync_index", "object_id", "keypoint_id"]):
        ports_in_group = sorted(group["cam_id"].unique())

        for i, port_a in enumerate(ports_in_group):
            for port_b in ports_in_group[i + 1 :]:
                pair = tuple(sorted((port_a, port_b)))
                if pair not in board_counts:
                    board_counts[pair] = set()
                board_counts[pair].add(sync_idx)

    # Convert to counts
    board_counts = {pair: len(boards) for pair, boards in board_counts.items()}
    sorted_boards = sorted(board_counts.items(), key=lambda x: x[1], reverse=True)

    print("\n  Port pairs that share boards (sorted by shared board count):")
    for (port_a, port_b), count in sorted_boards:
        print(f"    {port_a}-{port_b}: {count:,} shared boards")

    # 5. Summary for test expectations
    print("\n📋 SUMMARY FOR TEST EXPECTATIONS:")

    # Which cameras have ANY connectivity
    all_ports_with_data = set(df["cam_id"].unique())
    ports_with_pairs = set()
    for a, b in pair_counts.keys():
        ports_with_pairs.add(a)
        ports_with_pairs.add(b)

    print(f"\n  Cameras with observations: {sorted(all_ports_with_data)}")
    print(f"  Cameras with direct pairs: {sorted(ports_with_pairs)}")

    # Identify isolated cameras
    isolated = all_ports_with_data - ports_with_pairs
    if isolated:
        print(f"  ⚠️  Isolated cameras (no direct pairs): {sorted(isolated)}")
    else:
        print("  ✓ No isolated cameras")

    # 6. Return structured data for assertions
    connectivity = {
        "ports_with_data": sorted(all_ports_with_data),
        "direct_pairs": {
            pair: {"shared_points": count, "shared_boards": board_counts.get(pair, 0)} for pair, count in sorted_pairs
        },
        "isolated_ports": sorted(isolated),
    }

    print("\n" + "=" * 80)
    return connectivity


def test_missing_extrinsics(tmp_path: Path):
    version = "not_sufficient_stereopairs"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    xy_data_path = Path(tmp_path, "xy_CHARUCO.csv")
    camera_array = CameraArray.from_toml(tmp_path / "camera_array.toml")

    image_points = ImagePoints.from_csv(xy_data_path)

    # === RUN STEREO CALIBRATION WITH DEBUG LOGGING ===

    paired_pose_network: PairedPoseNetwork = build_paired_pose_network(image_points, camera_array)

    # === INSPECT THE GRAPH STATE ===
    print("\n" + "=" * 80)
    print("POST-CALIBRATION GRAPH INSPECTION")
    print("=" * 80)

    print(f"\nGraph contains {len(paired_pose_network._pairs)} pairs:")
    for (src, dst), pair in sorted(paired_pose_network._pairs.items()):
        print(f"  {src}→{dst}: RMSE={pair.error_score:.4f}")

    # Check for critical pairs
    critical_pairs = [(3, 4), (2, 4), (4, 3), (4, 2)]
    for src, dst in critical_pairs:
        pair = paired_pose_network.get_pair(src, dst)
        if pair:
            print(f"✓ Pair {src}→{dst} exists: RMSE={pair.error_score:.4f}")
        else:
            print(f"✗ Pair {src}→{dst} is MISSING")

    print("=" * 80)

    # === APPLY GRAPH AND LOG EVERY STEP ===
    paired_pose_network.apply_to(camera_array)
    logger.info("Camera Poses estimated from stereocalibration")

    # should have posed all cameras but 4 and 5 (4 is ignored)
    # Using set for posed_cameras to avoid order dependency
    assert set(camera_array.posed_cameras.keys()) == {1, 2, 3, 6}
    assert list(camera_array.unposed_cameras.keys()) == [4, 5]

    # camera 5 should not be in the index used for optimization parameter mapping
    assert 5 not in camera_array.posed_cam_id_to_index
    assert 4 not in camera_array.posed_cam_id_to_index

    # BundleParameterization pack/unpack round-trip leaves unposed cameras untouched
    from copy import deepcopy

    from caliscope.core.bundle_parameterization import BundleParameterization

    n_points = 10
    parameterization = BundleParameterization.from_camera_array(
        camera_array, n_points=n_points, refine_intrinsics=False
    )
    points = np.zeros((n_points, 3))
    x = parameterization.pack(camera_array, points)

    # Perturb all params slightly
    x += 0.01

    copy_array = deepcopy(camera_array)
    parameterization.unpack_into(copy_array, x)

    # Unposed cameras must remain untouched
    assert copy_array.cameras[5].rotation is None, "Unposed camera rotation should remain None"
    assert copy_array.cameras[5].translation is None, "Unposed camera translation should remain None"


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()

    temp_path = Path(__file__).parent / "debug"
    test_missing_extrinsics(temp_path)
    print("end")
    # test_deterministic_consistency()
