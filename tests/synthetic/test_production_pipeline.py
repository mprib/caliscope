"""Ground-truth pose accuracy through the production pipeline.

Every test drives a synthetic scene through calibrate_extrinsics() — the
function the extrinsic calibration presenter triggers — via
run_production_pipeline(), then asserts recovered camera poses against ground
truth. The existing tests/synthetic/*.py suite exercises the solver directly
(bootstrap + optimize); this file covers the production-only paths: blind
intrinsic synthesis, the two-phase robust loss, the static-marker guard, and
the filter + re-optimize sequence.

Tolerances are derived, not tuned. Each carries a one-line derivation citing
its precedent among the existing synthetic tests. A failure is a finding, not a
prompt to relax the bound.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from caliscope.core.aruco_marker import ArucoMarkerSet
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.charuco import Charuco
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import STATIC_SYNC_INDEX, ImagePoints
from caliscope.synthetic import SyntheticScene
from caliscope.synthetic.scene_factories import (
    charuco_target_scene,
    default_ring_scene,
    mirror_pair_ring_scene,
    mirror_pair_two_camera_scene,
    outlier_scene,
    wand_scene_with_constraints,
)
from tests.synthetic.assertions import align_to_ground_truth, pose_error
from tests.synthetic.production import ProductionRun, run_production_pipeline


@dataclass(frozen=True)
class Case:
    run: ProductionRun
    scene: SyntheticScene


@dataclass(frozen=True)
class FilterCase:
    run: ProductionRun
    baseline_max_translation_m: float


@dataclass(frozen=True)
class CharucoCase:
    run: ProductionRun
    scene: SyntheticScene
    constrained_rmse_mm: float
    unconstrained_rmse_mm: float


@dataclass(frozen=True)
class MirrorZeroThicknessCase:
    run: ProductionRun
    point_remap_count: int


@dataclass(frozen=True)
class MirrorThickBoardCase:
    run: ProductionRun
    marker_set: ArucoMarkerSet
    constraints: ConstraintSet


@dataclass(frozen=True)
class MirrorCoexistenceCase:
    run: ProductionRun
    marker_set: ArucoMarkerSet
    constraints: ConstraintSet


def _clean_case() -> Case:
    scene = default_ring_scene()
    return Case(run=run_production_pipeline(scene), scene=scene)


def _outlier_case() -> Case:
    scene, corrupted, _ = outlier_scene()
    return Case(run=run_production_pipeline(scene, image_points=corrupted), scene=scene)


def _filter_case() -> FilterCase:
    scene, corrupted, _ = outlier_scene()
    run = run_production_pipeline(scene, image_points=corrupted)

    # Baseline: a single linear solve on the corrupted data, no robust pass and
    # no filter — the accuracy floor the production pipeline must beat.
    baseline = CaptureVolume.bootstrap(corrupted, scene.intrinsics_only_cameras()).optimize()
    baseline_aligned = align_to_ground_truth(baseline, scene)
    baseline_max_t = max(
        pose_error(
            baseline_aligned.camera_array.cameras[cid],
            scene.camera_array.cameras[cid],
        ).translation_m
        for cid in scene.camera_array.posed_cameras
    )
    return FilterCase(run=run, baseline_max_translation_m=baseline_max_t)


def _rigid_case() -> Case:
    scene, constraints = wand_scene_with_constraints(include_static=True)
    return Case(run=run_production_pipeline(scene, constraints=constraints), scene=scene)


def _charuco_case() -> CharucoCase:
    scene = charuco_target_scene()

    # A production Charuco matching the scene's board (5x7 squares, 5cm). Its
    # getChessboardCorners() corner ids are the same ids the synthetic board
    # emits as keypoint_id — target_factories.charuco_board is built to mirror
    # that layout — so the compiled DistanceConstraints index exactly the points
    # the scene triangulates.
    charuco = Charuco.from_squares(columns=7, rows=5, square_size_cm=5.0)
    constraints = ConstraintSet.from_charuco(charuco)
    run = run_production_pipeline(scene, constraints=constraints)

    # Unconstrained baseline through the same pipeline. The constraints are
    # attached afterward only to *measure* rigidity on the result — they never
    # entered its optimization — mirroring TestConstrainedVsUnconstrainedOptimization.
    unconstrained = run_production_pipeline(scene, constraints=None)
    unconstrained_measured = replace(unconstrained.result.capture_volume, constraints=constraints)

    return CharucoCase(
        run=run,
        scene=scene,
        constrained_rmse_mm=run.result.capture_volume.rigidity_report().rmse_mm,
        unconstrained_rmse_mm=unconstrained_measured.rigidity_report().rmse_mm,
    )


def _blind_case() -> Case:
    scene, constraints = wand_scene_with_constraints(include_static=False)
    return Case(run=run_production_pipeline(scene, constraints=constraints, blind=True), scene=scene)


def _static_guard_case() -> Case:
    scene, constraints = wand_scene_with_constraints(include_static=True)

    # Corrupt one corner of static marker 3 (same shape-distorting approach as
    # TestStaticMarkerGuard in tests/test_calibrate_extrinsics.py). A uniform
    # offset on all four corners triangulates to a translated-but-rigid phantom
    # (~100mm RMSE, under the guard's ~106mm threshold); displacing a single
    # corner distorts the marker's shape to ~260mm and fires the guard.
    img_df = scene.image_points_noisy.df.copy()
    mask = (img_df["object_id"] == 3) & (img_df["keypoint_id"] == 0)
    img_df.loc[mask, "img_loc_x"] += 500.0
    img_df.loc[mask, "img_loc_y"] += 300.0
    corrupted = ImagePoints(img_df)

    return Case(
        run=run_production_pipeline(scene, image_points=corrupted, constraints=constraints),
        scene=scene,
    )


def _mirror_pair_zero_thickness_case() -> MirrorZeroThicknessCase:
    scene, marker_set = mirror_pair_two_camera_scene()
    constraints = ConstraintSet.from_marker_set(marker_set)
    run = run_production_pipeline(scene, constraints=constraints)
    return MirrorZeroThicknessCase(run=run, point_remap_count=len(constraints.point_remaps))


def _mirror_pair_thick_board_case() -> MirrorThickBoardCase:
    scene, marker_set = mirror_pair_ring_scene(thickness_m=0.005)
    constraints = ConstraintSet.from_marker_set(marker_set)
    run = run_production_pipeline(scene, constraints=constraints)
    return MirrorThickBoardCase(run=run, marker_set=marker_set, constraints=constraints)


def _mirror_pair_coexistence_case() -> MirrorCoexistenceCase:
    scene, marker_set = mirror_pair_ring_scene(thickness_m=0.005, with_anchored_marker=True)
    constraints = ConstraintSet.from_marker_set(marker_set)
    run = run_production_pipeline(scene, constraints=constraints)
    return MirrorCoexistenceCase(run=run, marker_set=marker_set, constraints=constraints)


@pytest.fixture(scope="module")
def clean_case() -> Case:
    return _clean_case()


@pytest.fixture(scope="module")
def outlier_case() -> Case:
    return _outlier_case()


@pytest.fixture(scope="module")
def filter_case() -> FilterCase:
    return _filter_case()


@pytest.fixture(scope="module")
def rigid_case() -> Case:
    return _rigid_case()


@pytest.fixture(scope="module")
def charuco_case() -> CharucoCase:
    return _charuco_case()


@pytest.fixture(scope="module")
def blind_case() -> Case:
    return _blind_case()


@pytest.fixture(scope="module")
def static_guard_case() -> Case:
    return _static_guard_case()


@pytest.fixture(scope="module")
def mirror_zero_thickness_case() -> MirrorZeroThicknessCase:
    return _mirror_pair_zero_thickness_case()


@pytest.fixture(scope="module")
def mirror_thick_board_case() -> MirrorThickBoardCase:
    return _mirror_pair_thick_board_case()


@pytest.fixture(scope="module")
def mirror_coexistence_case() -> MirrorCoexistenceCase:
    return _mirror_pair_coexistence_case()


class TestCleanSceneProduction:
    def test_pose_recovery(self, clean_case: Case) -> None:
        # 0.5 deg / 5mm: covariance propagation at 0.5px sigma, 4-cam ring,
        # 20 frames — same scene and bound as test_multistage_flow /
        # test_alignment_gauge. Production filters at 2.5 (vs the solver
        # flow's 5), removing less, so it cannot be worse.
        for cam_id, err in clean_case.run.pose_errors.items():
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"

    def test_result_fields(self, clean_case: Case) -> None:
        result = clean_case.run.result
        posed_ids = set(clean_case.scene.camera_array.posed_cameras)

        assert result.synthesized_cam_ids == frozenset()
        assert result.dropped_static_markers == ()
        assert result.bound_warnings == ()
        assert set(result.depth_ratios) == posed_ids
        # Ring geometry: depth ratio ~1.29 < 2.0 gate, so refinement is gated off.
        assert result.intrinsic_refinement_gated

        assert len(result.intrinsic_estimates) == len(posed_ids)
        for est in result.intrinsic_estimates:
            true_cam = clean_case.scene.camera_array.cameras[est.cam_id]
            assert true_cam.matrix is not None
            f_true = float(true_cam.matrix[0, 0])
            # Refinement was gated off, so f is untouched from the provided value:
            # recovered and initial both equal ground truth exactly.
            assert est.f_recovered == f_true
            assert est.f_initial == f_true


class TestOutlierProduction:
    def test_pose_recovery_with_outliers(self, outlier_case: Case) -> None:
        # 1.0 deg / 10mm: same bound as test_post_filter_poses_match_clean_baseline
        # (measured worst 0.11 deg / 5.4mm, 2x ceiling). Production adds the
        # soft_l1 pass before filtering, which can only help.
        for cam_id, err in outlier_case.run.pose_errors.items():
            assert err.rotation_deg < 1.0, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 1.0 deg"
            assert err.translation_m < 0.010, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 10 mm"

    def test_converged(self, outlier_case: Case) -> None:
        status = outlier_case.run.result.capture_volume.optimization_status
        assert status is not None
        assert status.converged


class TestFilterImprovement:
    def test_production_beats_single_pass(self, filter_case: FilterCase) -> None:
        # The relative assertion is the purpose of this test: filter +
        # re-optimize must not lose accuracy vs a bare single linear solve.
        assert filter_case.run.max_translation_m <= filter_case.baseline_max_translation_m, (
            f"production worst translation {filter_case.run.max_translation_m * 1000:.2f} mm > "
            f"single-pass baseline {filter_case.baseline_max_translation_m * 1000:.2f} mm"
        )
        # Absolute ceiling: same 1.0 deg / 10mm as TestOutlierProduction.
        assert filter_case.run.max_rotation_deg < 1.0, f"rotation {filter_case.run.max_rotation_deg:.3f} deg > 1.0 deg"
        assert filter_case.run.max_translation_m < 0.010, (
            f"translation {filter_case.run.max_translation_m * 1000:.2f} mm > 10 mm"
        )


class TestRigidConstraintsProduction:
    def test_pose_recovery(self, rigid_case: Case) -> None:
        # 0.5 deg / 5mm: 4-cam ring at radius 1.2 (tighter than the 2.0m
        # default ring) and 40 frames (2x the default ring's observations),
        # both at 0.5px sigma. Both factors lower covariance, so the
        # default-ring ceiling holds a fortiori.
        for cam_id, err in rigid_case.run.pose_errors.items():
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"

    def test_rigidity(self, rigid_case: Case) -> None:
        # < 2.0mm: precedent TestEndToEndBlind in tests/test_calibrate_extrinsics.py.
        rmse_mm = rigid_case.run.result.capture_volume.rigidity_report().rmse_mm
        assert rmse_mm < 2.0, f"rigidity RMSE {rmse_mm:.3f}mm > 2.0mm"

    def test_static_markers_survive(self, rigid_case: Case) -> None:
        # Wand geometry: depth ratio >= 2.25 >= 2.0 gate, so refinement runs.
        assert not rigid_case.run.result.intrinsic_refinement_gated
        assert rigid_case.run.result.dropped_static_markers == ()
        world_df = rigid_case.run.result.capture_volume.world_points.df
        for marker_id in (2, 3, 4, 5):
            rows = world_df[(world_df["object_id"] == marker_id) & (world_df["sync_index"] == STATIC_SYNC_INDEX)]
            assert len(rows) == 4, f"static marker {marker_id}: {len(rows)} world rows, expected 4"

    def test_centroid_constraints_consumed(self, rigid_case: Case) -> None:
        # Guard against a silently-empty centroid path: the ConstraintSet the
        # pipeline actually optimized must still carry both center links — the
        # mobile wand center (markers 0-1) and the static-static wall-marker
        # center link (markers 2-4). The migrated wand fixture is the only
        # scene exercising CentroidDistanceConstraint end to end.
        consumed = rigid_case.run.result.capture_volume.constraints
        assert consumed is not None
        centroid_pairs = {frozenset((c.object_id_a, c.object_id_b)) for c in consumed.centroid_distances}
        assert frozenset((0, 1)) in centroid_pairs, f"missing wand center link; have {centroid_pairs}"
        assert frozenset((2, 4)) in centroid_pairs, f"missing static-static center link; have {centroid_pairs}"

    def test_centroid_center_link_fires(self, rigid_case: Case) -> None:
        # The static wall-marker center link (markers 2-4) is the long-lever
        # tape-measure scale anchor. It must survive the full pipeline and be
        # evaluated once at STATIC_SYNC_INDEX — proving the centroid path is not
        # silently skipped. Its magnitude is already bounded by the pooled
        # rigidity RMSE in test_rigidity; here we assert only that it fired.
        report = rigid_case.run.result.capture_volume.rigidity_report()
        static_center = [
            v for v in report.violations if v.kind == "centroid" and {v.object_id_a, v.object_id_b} == {2, 4}
        ]
        assert len(static_center) == 1, f"expected one static-static centroid violation, got {len(static_center)}"
        v = static_center[0]
        assert v.sync_index == STATIC_SYNC_INDEX
        assert v.keypoint_id_a == -1 and v.keypoint_id_b == -1  # centroids name no single corner


class TestCharucoConstraintsProduction:
    def test_pose_recovery(self, charuco_case: CharucoCase) -> None:
        # 0.5 deg / 5mm: charuco_target_scene is the default 4-cam ring
        # (radius 2.0, height 0.5), 20 frames, 0.5px sigma — identical geometry
        # to default_ring_scene, only the target differs (charuco board vs
        # planar grid). Same covariance-propagation derivation as
        # TestCleanSceneProduction.test_pose_recovery holds a fortiori: the
        # board geometry now also anchors BA as distance constraints.
        for cam_id, err in charuco_case.run.pose_errors.items():
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"

    def test_rigidity(self, charuco_case: CharucoCase) -> None:
        # < 2.0mm: same bound and precedent as TestRigidConstraintsProduction
        # (TestEndToEndBlind in tests/test_calibrate_extrinsics.py).
        assert charuco_case.constrained_rmse_mm < 2.0, f"rigidity RMSE {charuco_case.constrained_rmse_mm:.3f}mm > 2.0mm"

    def test_constrained_beats_unconstrained(self, charuco_case: CharucoCase) -> None:
        # Precedent TestConstrainedVsUnconstrainedOptimization: adding the board
        # geometry as constraints cannot worsen rigidity. The relative bound is
        # the point of the test; the absolute ceiling lives in test_rigidity.
        assert charuco_case.constrained_rmse_mm <= charuco_case.unconstrained_rmse_mm, (
            f"constrained RMSE {charuco_case.constrained_rmse_mm:.3f}mm > "
            f"unconstrained RMSE {charuco_case.unconstrained_rmse_mm:.3f}mm"
        )


class TestMirrorPairZeroThickness:
    def test_both_cameras_posed(self, mirror_zero_thickness_case: MirrorZeroThicknessCase) -> None:
        posed = set(mirror_zero_thickness_case.run.result.capture_volume.camera_array.posed_cameras)
        assert posed == {0, 1}

    def test_point_remaps_compiled(self, mirror_zero_thickness_case: MirrorZeroThicknessCase) -> None:
        # Four remaps: one per corner of the zero-thickness pair, merging face
        # B's observations onto face A's identity.
        assert mirror_zero_thickness_case.point_remap_count == 4

    def test_pose_recovery(self, mirror_zero_thickness_case: MirrorZeroThicknessCase) -> None:
        # 1.0 deg / 20mm: two facing cameras merging one planar marker face
        # each via the zero-thickness remap — minimal redundancy, sitting near
        # the two-fold planar-pose ambiguity that the factory's tilt-sweep
        # trajectory breaks. Measured worst across seeds {42,7,99,123,2024}:
        # 0.61 deg / 13.0 mm (seed 123). The 1.0 deg / 20 mm bound is ~1.5x the
        # worst seed, the looser derived bound the plan anticipated for the
        # two-camera minimal-redundancy case. Precedent: weak-geometry
        # synthetic scenes (narrow_baseline_scene, test_planar_degeneracy).
        for cam_id, err in mirror_zero_thickness_case.run.pose_errors.items():
            assert err.rotation_deg < 1.0, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 1.0 deg"
            assert err.translation_m < 0.020, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 20 mm"


class TestMirrorPairThickBoard:
    def test_all_cameras_posed(self, mirror_thick_board_case: MirrorThickBoardCase) -> None:
        posed = set(mirror_thick_board_case.run.result.capture_volume.camera_array.posed_cameras)
        assert posed == {0, 1, 2, 3}

    def test_thickness_constraints_compiled(self, mirror_thick_board_case: MirrorThickBoardCase) -> None:
        pair = mirror_thick_board_case.marker_set.mirror_pairs[0]
        thickness_constraints = [
            c for c in mirror_thick_board_case.constraints.distances if {c.object_id_a, c.object_id_b} == {0, 1}
        ]
        assert len(thickness_constraints) == 4
        for c in thickness_constraints:
            assert c.distance == pytest.approx(0.005)
            assert c.sigma == pytest.approx(0.002)
        corner_pairs = {(c.keypoint_id_a, c.keypoint_id_b) for c in thickness_constraints}
        assert corner_pairs == set(pair.corner_mapping)

    def test_pose_recovery(self, mirror_thick_board_case: MirrorThickBoardCase) -> None:
        # 0.5 deg / 5mm: standard 4-camera ring covariance-propagation bound,
        # identical to TestRigidConstraintsProduction. Measured 0.07 deg /
        # 0.63 mm.
        for cam_id, err in mirror_thick_board_case.run.pose_errors.items():
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"

    def test_rigidity(self, mirror_thick_board_case: MirrorThickBoardCase) -> None:
        # < 2.0mm: precedent TestEndToEndBlind in tests/test_calibrate_extrinsics.py,
        # same as TestRigidConstraintsProduction.test_rigidity. Measured 0.39 mm.
        rmse_mm = mirror_thick_board_case.run.result.capture_volume.rigidity_report().rmse_mm
        assert rmse_mm < 2.0, f"rigidity RMSE {rmse_mm:.3f}mm > 2.0mm"


class TestMirrorPairCoexistence:
    def test_constraints_compiled(self, mirror_coexistence_case: MirrorCoexistenceCase) -> None:
        constraints = mirror_coexistence_case.constraints
        thickness_constraints = [c for c in constraints.distances if {c.object_id_a, c.object_id_b} == {0, 1}]
        assert len(thickness_constraints) == 4

        center_links = [c for c in constraints.centroid_distances if {c.object_id_a, c.object_id_b} == {0, 2}]
        assert len(center_links) == 1

    def test_center_link_fires(self, mirror_coexistence_case: MirrorCoexistenceCase) -> None:
        # Measured: it fires on 23 frames.
        report = mirror_coexistence_case.run.result.capture_volume.rigidity_report()
        center_violations = [
            v for v in report.violations if v.kind == "centroid" and {v.object_id_a, v.object_id_b} == {0, 2}
        ]
        assert len(center_violations) >= 1, "expected the 0-2 center link to fire at least one violation"

    def test_pose_recovery(self, mirror_coexistence_case: MirrorCoexistenceCase) -> None:
        # 0.5 deg / 5mm: same ring bound as TestMirrorPairThickBoard — the
        # coexisting center link to a third marker does not degrade geometry.
        posed = set(mirror_coexistence_case.run.result.capture_volume.camera_array.posed_cameras)
        assert posed == {0, 1, 2, 3}
        for cam_id, err in mirror_coexistence_case.run.pose_errors.items():
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"


class TestBlindIntrinsicsProduction:
    def test_synthesized_ids(self, blind_case: Case) -> None:
        assert blind_case.run.result.synthesized_cam_ids == frozenset(blind_case.scene.camera_array.cameras)

    def test_f_recovery(self, blind_case: Case) -> None:
        # < 2% per camera: precedent TestEndToEndBlind.
        for est in blind_case.run.result.intrinsic_estimates:
            true_cam = blind_case.scene.camera_array.cameras[est.cam_id]
            assert true_cam.matrix is not None
            f_true = float(true_cam.matrix[0, 0])
            f_err_pct = abs(est.f_recovered - f_true) / f_true * 100
            assert f_err_pct < 2.0, f"cam {est.cam_id}: f error {f_err_pct:.2f}%"

    def test_pose_recovery(self, blind_case: Case) -> None:
        # 1.0 deg rotation; 16mm translation. Translation ceiling is a fitted
        # blind-coupling bound (5-seed fit, tests/tmp/exp3_results.txt): base
        # 9.2mm + kappa(0.117) x max-allowed f error (2%) x camera radius
        # (1200mm) = 2.8mm + 3 x scatter RMSE (3 x 1.29mm) = 3.9mm -> 15.9 ~ 16mm.
        # Measured worst across seeds 42/7/99/123/2024: 9.7-12.8mm. Replaces the
        # old undocumented 2x convention with a bound at the f tolerance this
        # test itself permits.
        for cam_id, err in blind_case.run.pose_errors.items():
            assert err.rotation_deg < 1.0, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 1.0 deg"
            assert err.translation_m < 0.016, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 16 mm"


class TestStaticMarkerGuardProduction:
    def test_dropped(self, static_guard_case: Case) -> None:
        assert 3 in static_guard_case.run.result.dropped_static_markers
        world_df = static_guard_case.run.result.capture_volume.world_points.df
        assert (world_df["object_id"] == 3).sum() == 0

    def test_survivors_unharmed(self, static_guard_case: Case) -> None:
        # The guard's whole point: a bad static marker doesn't poison the
        # solution. Survivors hold the clean 0.5 deg / 5mm rigid-scene bound.
        for cam_id, err in static_guard_case.run.pose_errors.items():
            assert err.rotation_deg < 0.5, f"cam {cam_id}: rotation {err.rotation_deg:.3f} deg > 0.5 deg"
            assert err.translation_m < 0.005, f"cam {cam_id}: translation {err.translation_m * 1000:.2f} mm > 5 mm"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Clean scene...")
    clean = _clean_case()
    TestCleanSceneProduction().test_pose_recovery(clean)
    TestCleanSceneProduction().test_result_fields(clean)
    print("  PASSED")

    print("Outliers...")
    outlier = _outlier_case()
    TestOutlierProduction().test_pose_recovery_with_outliers(outlier)
    TestOutlierProduction().test_converged(outlier)
    print("  PASSED")

    print("Filter improvement...")
    flt = _filter_case()
    TestFilterImprovement().test_production_beats_single_pass(flt)
    print("  PASSED")

    print("Rigid constraints...")
    rigid = _rigid_case()
    TestRigidConstraintsProduction().test_pose_recovery(rigid)
    TestRigidConstraintsProduction().test_rigidity(rigid)
    TestRigidConstraintsProduction().test_static_markers_survive(rigid)
    TestRigidConstraintsProduction().test_centroid_constraints_consumed(rigid)
    TestRigidConstraintsProduction().test_centroid_center_link_fires(rigid)
    print("  PASSED")

    print("Charuco constraints...")
    charuco = _charuco_case()
    TestCharucoConstraintsProduction().test_pose_recovery(charuco)
    TestCharucoConstraintsProduction().test_rigidity(charuco)
    TestCharucoConstraintsProduction().test_constrained_beats_unconstrained(charuco)
    print("  PASSED")

    print("Blind intrinsics...")
    blind = _blind_case()
    TestBlindIntrinsicsProduction().test_synthesized_ids(blind)
    TestBlindIntrinsicsProduction().test_f_recovery(blind)
    TestBlindIntrinsicsProduction().test_pose_recovery(blind)
    print("  PASSED")

    print("Static marker guard...")
    guard = _static_guard_case()
    TestStaticMarkerGuardProduction().test_dropped(guard)
    TestStaticMarkerGuardProduction().test_survivors_unharmed(guard)
    print("  PASSED")

    print("Mirror pair zero-thickness...")
    mirror_zero = _mirror_pair_zero_thickness_case()
    TestMirrorPairZeroThickness().test_both_cameras_posed(mirror_zero)
    TestMirrorPairZeroThickness().test_point_remaps_compiled(mirror_zero)
    TestMirrorPairZeroThickness().test_pose_recovery(mirror_zero)
    print("  PASSED")

    print("Mirror pair thick board...")
    mirror_thick = _mirror_pair_thick_board_case()
    TestMirrorPairThickBoard().test_all_cameras_posed(mirror_thick)
    TestMirrorPairThickBoard().test_thickness_constraints_compiled(mirror_thick)
    TestMirrorPairThickBoard().test_pose_recovery(mirror_thick)
    TestMirrorPairThickBoard().test_rigidity(mirror_thick)
    print("  PASSED")

    print("Mirror pair coexistence...")
    mirror_coexist = _mirror_pair_coexistence_case()
    TestMirrorPairCoexistence().test_constraints_compiled(mirror_coexist)
    TestMirrorPairCoexistence().test_center_link_fires(mirror_coexist)
    TestMirrorPairCoexistence().test_pose_recovery(mirror_coexist)
    print("  PASSED")
