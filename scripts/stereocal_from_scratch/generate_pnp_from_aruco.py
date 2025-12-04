"""
PnP-based Relative Pose Estimation for Camera Array Initialization.

This script provides a clean, modular workflow for estimating camera pair extrinsics
using 3D object keypoints and PnP, as an alternative to cv2.stereocalibrate. The
resulting poses are packaged into a PairedPoseNetwork for compatibility with the
existing calibration pipeline.

Design Decisions:
- Uses SOLVEPNP_IPPE for planar targets (optimal) with iterative fallback
- Undistorts points once at the PnP stage to work in normalized coordinates
- Applies IQR-based outlier rejection to translation magnitude and rotation angle
- Aggregates poses via quaternion averaging (robust to rotation space non-linearity)
- Validates against gold standard using rotation angle and translation errors
- Computes Stereo RMSE via triangulation/reprojection on the normalized plane
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from caliscope import __root__
from caliscope.calibration.array_initialization.paired_pose_network import PairedPoseNetwork
from caliscope.calibration.array_initialization.pose_network_builder import (
    PoseNetworkBuilder,
    rotation_error,
    translation_error,
)
from caliscope.calibration.array_initialization.estimate_paired_pose_network import (
    estimate_paired_pose_network,
)
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.configurator import Configurator
from caliscope.logger import setup_logging
from caliscope.post_processing.point_data import ImagePoints

setup_logging()
logger = logging.getLogger(__name__)


def compare_to_gold_standard(
    pnp_network: PairedPoseNetwork, gold_standard_network: PairedPoseNetwork, output_dir: Path
) -> pd.DataFrame:
    """
    Compare PnP-based PairedPoseNetwork to gold standard and generate metrics.

    Returns:
        DataFrame with comparison metrics for each pair
    """
    logger.info("Comparing to gold standard...")

    comparison_rows = []
    for pair, pnp_pair in pnp_network._pairs.items():
        if pair not in gold_standard_network._pairs:
            logger.warning(f"Gold standard not found for pair {pair}")
            continue

        gs_pair = gold_standard_network._pairs[pair]

        rot_err = rotation_error(pnp_pair.rotation, gs_pair.rotation)
        trans_err = translation_error(pnp_pair.translation, gs_pair.translation)

        comparison_rows.append(
            {
                "pair": f"stereo_{pair[0]}_{pair[1]}",
                "port_a": pair[0],
                "port_b": pair[1],
                "rotation_delta_deg": rot_err,
                "translation_magnitude_delta_pct": trans_err["magnitude_delta_pct"],
                "translation_direction_delta_deg": trans_err["direction_delta_deg"],
                "pnp_translation_norm": np.linalg.norm(pnp_pair.translation),
                "gold_translation_norm": np.linalg.norm(gs_pair.translation),
                "relative_translation_diff": np.linalg.norm(pnp_pair.translation - gs_pair.translation),
                "pnp_rmse": pnp_pair.error_score,
                "gold_rmse": gs_pair.error_score,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    output_file = output_dir / "comparison_table.csv"
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"Comparison table saved to {output_file}")

    # Log summary statistics
    logger.info("=" * 50)
    logger.info("VALIDATION SUMMARY: COMPARISON WITH GOLD STANDARD")
    logger.info("=" * 50)
    logger.info(f"Mean rotation delta {comparison_df['rotation_delta_deg'].mean():.4f}°")
    logger.info(f"Std rotation delta {comparison_df['rotation_delta_deg'].std():.4f}°")
    logger.info(f"Mean translation magnitude delta {comparison_df['translation_magnitude_delta_pct'].mean():.2f}%")
    logger.info(f"Mean translation direction delta {comparison_df['translation_direction_delta_deg'].mean():.4f}°")
    logger.info(f"Max rotation delta {comparison_df['rotation_delta_deg'].max():.4f}°")
    logger.info(f"Max translation magnitude delta {comparison_df['translation_magnitude_delta_pct'].max():.2f}%")

    return comparison_df


def save_network_to_json(network: PairedPoseNetwork, output_path: Path) -> None:
    """Serialize a PairedPoseNetwork to JSON for inspection/debugging."""
    data = {}
    for pair, stereo_pair in network._pairs.items():
        data[f"stereo_{pair[0]}_{pair[1]}"] = {
            "primary_port": stereo_pair.primary_port,
            "secondary_port": stereo_pair.secondary_port,
            "error_score": stereo_pair.error_score,
            "rotation": stereo_pair.rotation.tolist(),
            "translation": stereo_pair.translation.tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Network saved to {output_path}")


def main():
    """Main validation pipeline comparing PnP to stereocalibrate gold standard."""
    logger.info("Starting PnP validation pipeline...")

    script_dir = Path(__file__).parent
    output_dir = script_dir / "working_output"
    output_dir.mkdir(exist_ok=True)

    project_fixture_dir = __root__ / "scripts/stereocal_from_scratch/aruco_pipeline"
    calibration_video_dir = project_fixture_dir / "calibration/extrinsic"
    charuco_point_data_file = calibration_video_dir / "CHARUCO/xy_CHARUCO.csv"

    config = Configurator(project_fixture_dir)
    camera_array = config.get_camera_array()
    point_data = pd.read_csv(charuco_point_data_file)

    # Stage 1: Generate gold standard using legacy stereocalibrate
    logger.info("=" * 20 + " STAGE 1: Gold Standard Generation " + "=" * 20)
    image_points = ImagePoints(point_data)
    gold_standard_network = estimate_paired_pose_network(image_points, camera_array, boards_sampled=10)
    save_network_to_json(gold_standard_network, output_dir / "gold_standard.json")

    # Stage 2: Generate PnP-based pose network using NEW BUILDER API
    logger.info("=" * 20 + " STAGE 2: PnP Pose Network Generation (Builder API) " + "=" * 20)

    # Example 1: Default configuration
    builder = PoseNetworkBuilder(camera_array, image_points)
    pnp_network = builder.estimate_camera_to_object_poses().estimate_relative_poses().filter_outliers().build()

    # Example 2: Custom configuration (demonstrating parameter options)
    builder_custom = PoseNetworkBuilder(camera_array, image_points)
    pnp_network_custom = (
        builder_custom.estimate_camera_to_object_poses(min_points=6, pnp_flags=cv2.SOLVEPNP_ITERATIVE)
        .estimate_relative_poses()
        .filter_outliers(threshold=2.0, rotation_threshold=5.0)  # Stricter rotation filter
        .build()
    )

    # Save both networks
    save_network_to_json(pnp_network, output_dir / "pnp_estimates.json")
    save_network_to_json(pnp_network_custom, output_dir / "pnp_estimates_custom.json")

    # Demonstrate apply_to() method
    logger.info("Applying PnP network to camera array...")
    test_array = config.get_camera_array()  # Fresh copy
    pnp_network.apply_to(test_array)
    logger.info(f"Updated poses for cameras: {list(test_array.posed_cameras.keys())}")

    # Stage 3: Validate against gold standard
    logger.info("=" * 20 + " STAGE 3: Gold Standard Comparison " + "=" * 20)
    compare_to_gold_standard(pnp_network, gold_standard_network, output_dir)

    # Also compare custom configuration
    logger.info("Comparing custom configuration...")
    compare_to_gold_standard(pnp_network_custom, gold_standard_network, output_dir / "custom_comparison")

    logger.info("=" * 20 + " VALIDATION COMPLETE " + "=" * 20)
    logger.info(f"Results saved to {output_dir}")

    logger.info("Test actual bundle adjustment")

    world_points = image_points.triangulate(test_array)

    capture_volume = CaptureVolume(test_array, world_points.to_point_estimates())
    capture_volume.optimize()


if __name__ == "__main__":
    main()
