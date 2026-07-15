"""Epipolar bootstrap (2D-only, no object geometry) through the production pipeline.

The epipolar path recovers extrinsics from 2D-2D correspondences via the
essential matrix, for trackers that emit only image points (body keypoints) with
no known object geometry. These tests drive it through the real
calibrate_extrinsics() seam with obj_loc nulled -- the same production entry
point the PnP tests use -- plus a couple of cheap unit tests pinning the two
convention-sensitive numerics (essential recovery, correspondence pooling).

Why a box (3D) target rather than the planar grid: the essential-matrix
decomposition is degenerate for coplanar points. The planar grid, with its
one-sided visibility culling, leaves each camera pair co-observing a
near-coplanar arc, so the essential estimate flips ~140 deg for a minority of
noise seeds -- a genuine geometric degeneracy, not a solver bug. The epipolar
path's real target is body keypoints: a 3D constellation seen from all sides.
box_target_scene is the faithful analog (non-planar, no visibility culling) and
recovers robustly across seeds.

Tolerances are derived from measured worst-across-seeds error with headroom,
against the framework's covariance-propagation ceiling (tests/synthetic/README.md:
translation ~ GEOMETRY_FACTOR x pixel_sigma ~ 7.5-10mm at 0.5px). Measured worst
over seeds 0-19: 4-cam 0.070 deg / 2.1 mm; over seeds 0-11: 2-cam 0.13 deg /
5.3 mm. The bounds below sit ~3-4x above the measured worst and within the
theory ceiling. A failure is a finding, not a prompt to relax the bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import pytest

from caliscope.cameras.camera_array import CameraData
from caliscope.core.bootstrap_pose.epipolar_pose_builder import (
    pooled_correspondences,
    recover_pair_pose,
)
from caliscope.core.calibrate_extrinsics import calibrate_extrinsics
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.point_data import ImagePoints
from caliscope.exceptions import CalibrationError
from caliscope.synthetic import (
    CameraSynthesizer,
    SyntheticScene,
    Trajectory,
    strip_intrinsics,
)
from caliscope.synthetic.scene_factories import box_target_scene
from caliscope.synthetic.target_factories import box_target
from tests.synthetic.production import ProductionRun, run_production_pipeline


def _null_obj_loc(image_points: ImagePoints) -> ImagePoints:
    """Strip object geometry, forcing the epipolar (2D-only) bootstrap path."""
    df = image_points.df.copy()
    df[["obj_loc_x", "obj_loc_y", "obj_loc_z"]] = np.nan
    return ImagePoints(df)


def _two_camera_box_scene(random_seed: int = 42) -> SyntheticScene:
    """A single stereo pair viewing the non-planar box target."""
    camera_array = CameraSynthesizer().add_ring(n=2, radius=2.0, height=0.5).build()
    calibration_object = box_target(width=0.4, height=0.4, depth=0.4)
    trajectory = Trajectory.orbital(n_frames=20, radius=0.2, arc_extent_deg=360.0, tumble_rate=1.0)
    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=0.5,
        random_seed=random_seed,
    )


@dataclass(frozen=True)
class Case:
    run: ProductionRun
    scene: SyntheticScene


def _four_cam_case() -> Case:
    scene = box_target_scene(random_seed=42)
    run = run_production_pipeline(scene, image_points=_null_obj_loc(scene.image_points_noisy))
    return Case(run=run, scene=scene)


def _two_cam_case() -> Case:
    scene = _two_camera_box_scene(random_seed=42)
    run = run_production_pipeline(scene, image_points=_null_obj_loc(scene.image_points_noisy))
    return Case(run=run, scene=scene)


@pytest.fixture(scope="module")
def four_cam_case() -> Case:
    return _four_cam_case()


@pytest.fixture(scope="module")
def two_cam_case() -> Case:
    return _two_cam_case()


class TestFourCameraEpipolar:
    def test_all_cameras_posed(self, four_cam_case: Case) -> None:
        posed = set(four_cam_case.run.result.capture_volume.camera_array.posed_cameras)
        assert posed == {0, 1, 2, 3}

    def test_pose_recovery(self, four_cam_case: Case) -> None:
        # 0.5 deg / 8 mm: 4-cam ring, 0.5px sigma. Rotation uses the scene's
        # standard covariance-propagation bound (as TestCleanSceneProduction);
        # translation is set at ~4x the measured worst-across-seeds (2.1mm) and
        # within the README's GEOMETRY_FACTOR ceiling (~7.5-10mm at 0.5px). The
        # epipolar path carries the looser translation bound the plan anticipated
        # for a bootstrap that starts from 2D-2D data with no metric anchor.
        for cam_id, err in four_cam_case.run.pose_errors.items():
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.008, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 8 mm"

    def test_intrinsics_untouched(self, four_cam_case: Case) -> None:
        # Real intrinsics were supplied (not blind), so nothing was synthesized.
        # The box's depth ratio (~1.2) is below the 2.0 refinement gate, so
        # intrinsics are left exactly as provided.
        result = four_cam_case.run.result
        assert result.synthesized_cam_ids == frozenset()
        assert result.intrinsic_refinement_gated
        for est in result.intrinsic_estimates:
            assert est.f_recovered == est.f_initial


class TestTwoCameraEpipolar:
    def test_both_cameras_posed(self, two_cam_case: Case) -> None:
        posed = set(two_cam_case.run.result.capture_volume.camera_array.posed_cameras)
        assert posed == {0, 1}

    def test_pose_recovery(self, two_cam_case: Case) -> None:
        # 0.5 deg / 10 mm: a single stereo pair has no third view for redundancy,
        # so its covariance is higher than the 4-cam ring's. Measured worst over
        # seeds 0-11 was 0.13 deg / 5.3mm; the bound is ~2x that and still under
        # the two-camera minimal-redundancy precedent (TestMirrorPairZeroThickness,
        # 20mm). Baseline scale is arbitrary; the Procrustes alignment absorbs it.
        for cam_id, err in two_cam_case.run.pose_errors.items():
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.010, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 10 mm"


class TestEpipolarGates:
    def test_uncalibrated_intrinsics_raises(self) -> None:
        # Blind intrinsics (f=width/2) are geometrically fatal for the essential
        # path: with no obj_loc anchor to absorb focal error, the decomposition
        # yields wrong poses. calibrate_extrinsics must refuse before bootstrap.
        scene = box_target_scene(random_seed=42)
        blind_cameras = strip_intrinsics(scene.intrinsics_only_cameras())
        with pytest.raises(CalibrationError, match="intrinsic"):
            calibrate_extrinsics(_null_obj_loc(scene.image_points_noisy), blind_cameras, None)

    def test_insufficient_overlap_raises(self) -> None:
        # Cameras that never co-observe the subject (disjoint frame ranges) share
        # zero correspondences, so no essential matrix can be formed.
        scene = _two_camera_box_scene(random_seed=42)
        df = _null_obj_loc(scene.image_points_noisy).df
        disjoint = df[
            ((df["cam_id"] == 0) & (df["sync_index"] < 10)) | ((df["cam_id"] == 1) & (df["sync_index"] >= 10))
        ].reset_index(drop=True)
        with pytest.raises(CalibrationError, match="overlap"):
            CaptureVolume.bootstrap(ImagePoints(disjoint), scene.intrinsics_only_cameras())


class TestEpipolarUnit:
    def test_recover_pair_pose_exact(self) -> None:
        # Noiseless recovery pins the convention-sensitive essential -> pose step:
        # rotation is exact, translation is recovered up to a sign-fixed unit
        # direction (essential geometry has no baseline scale).
        rng = np.random.default_rng(0)
        points_world = rng.uniform([-0.5, -0.5, 4.0], [0.5, 0.5, 6.0], size=(300, 3))
        matrix = np.array([[1600.0, 0, 960.0], [0, 1600.0, 540.0], [0, 0, 1.0]])
        cam_a = CameraData(cam_id=0, size=(1920, 1080), matrix=matrix, distortions=np.zeros(5))
        cam_b = CameraData(cam_id=1, size=(1920, 1080), matrix=matrix, distortions=np.zeros(5))

        rot_b = cv2.Rodrigues(np.array([0.05, 0.35, -0.1]))[0]
        t_b = np.array([1.2, 0.1, 0.3])

        pix_a = _project(points_world, np.eye(3), np.zeros(3), matrix)
        pix_b = _project(points_world, rot_b, t_b, matrix)

        pose = recover_pair_pose(pix_a, pix_b, camera_a=cam_a, camera_b=cam_b)
        # 1e-4: noiseless recovery is exact up to the RANSAC/recoverPose numerical
        # floor (~3e-5 here); the translation is a sign-fixed unit direction.
        assert np.allclose(pose["rotation"], rot_b, atol=1e-4)
        assert np.allclose(pose["translation"], t_b / np.linalg.norm(t_b), atol=1e-4)
        assert pose["conditioning"] > 0.9  # well-conditioned 3D cloud

    def test_pooled_correspondences_matches_and_drops_nan(self) -> None:
        # Shared (object_id, keypoint_id, sync_index) rows pair up; a correspondence
        # with a NaN pixel in either camera drops from the pool.
        df_a = pd.DataFrame(
            {
                "sync_index": [0, 0, 1, 2],
                "cam_id": [0, 0, 0, 0],
                "object_id": [0, 0, 0, 0],
                "keypoint_id": [1, 2, 1, 1],
                "img_loc_x": [10.0, 20.0, 11.0, np.nan],
                "img_loc_y": [10.0, 20.0, 11.0, 40.0],
            }
        )
        df_b = pd.DataFrame(
            {
                "sync_index": [0, 0, 1, 2],
                "cam_id": [1, 1, 1, 1],
                "object_id": [0, 0, 0, 0],
                "keypoint_id": [1, 2, 3, 1],  # (1,3) has no match in a; sync 2 kp 1 matches but a's pixel is NaN
                "img_loc_x": [110.0, 120.0, 130.0, 140.0],
                "img_loc_y": [110.0, 120.0, 130.0, 140.0],
            }
        )
        keys, pix_a, pix_b = pooled_correspondences(df_a, df_b)
        # Matches: (obj0,kp1,sync0), (obj0,kp2,sync0). (kp1,sync1) has no b match;
        # (kp1,sync2) is dropped for a's NaN pixel.
        assert len(keys) == 2
        matched = {(int(o), int(k), int(s)) for o, k, s in keys}
        assert matched == {(0, 1, 0), (0, 2, 0)}
        assert np.isfinite(pix_a).all() and np.isfinite(pix_b).all()


def _project(points_world: np.ndarray, rotation: np.ndarray, translation: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Pixel projection of world points under a world-to-camera pose (no distortion)."""
    cam = points_world @ rotation.T + translation
    image = cam @ matrix.T
    return image[:, :2] / image[:, 2:3]


if __name__ == "__main__":
    print("Four-camera epipolar...")
    four = _four_cam_case()
    TestFourCameraEpipolar().test_all_cameras_posed(four)
    TestFourCameraEpipolar().test_pose_recovery(four)
    TestFourCameraEpipolar().test_intrinsics_untouched(four)
    print(f"  worst rot {four.run.max_rotation_deg:.4f} deg, trans {four.run.max_translation_m * 1000:.3f} mm")

    print("Two-camera epipolar...")
    two = _two_cam_case()
    TestTwoCameraEpipolar().test_both_cameras_posed(two)
    TestTwoCameraEpipolar().test_pose_recovery(two)
    print(f"  worst rot {two.run.max_rotation_deg:.4f} deg, trans {two.run.max_translation_m * 1000:.3f} mm")

    print("Gates...")
    TestEpipolarGates().test_uncalibrated_intrinsics_raises()
    TestEpipolarGates().test_insufficient_overlap_raises()

    print("Unit...")
    TestEpipolarUnit().test_recover_pair_pose_exact()
    TestEpipolarUnit().test_pooled_correspondences_matches_and_drops_nan()
    print("  PASSED")
