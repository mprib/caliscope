"""Tests for lens profiles and intrinsic perturbation."""

import numpy as np
import pytest

from caliscope.synthetic.camera_synthesizer import (
    IDEAL,
    MACHINE_VISION,
    WEBCAM,
    CameraSynthesizer,
    IntrinsicPerturbation,
    LensProfile,
    perturb_intrinsics,
)


class TestLensProfiles:
    def test_default_is_webcam(self) -> None:
        array = CameraSynthesizer().add_ring(n=4, radius=2.0).build()
        for cam in array.cameras.values():
            assert cam.matrix is not None
            assert cam.matrix[0, 0] == pytest.approx(1394.6)
            assert cam.matrix[1, 1] == pytest.approx(1394.6)

    def test_ideal_profile(self) -> None:
        array = CameraSynthesizer().add_ring(n=2, radius=2.0, lens=IDEAL).build()
        for cam in array.cameras.values():
            assert cam.matrix is not None
            assert cam.matrix[0, 0] == pytest.approx(800.0)
            assert cam.distortions is not None
            np.testing.assert_array_equal(cam.distortions, np.zeros(5))

    def test_machine_vision_profile(self) -> None:
        array = CameraSynthesizer().add_ring(n=2, radius=2.0, lens=MACHINE_VISION).build()
        for cam in array.cameras.values():
            assert cam.matrix is not None
            assert cam.matrix[0, 0] == pytest.approx(1100.0)
            assert cam.distortions is not None
            assert cam.distortions[0] == pytest.approx(-0.37)

    def test_mixed_lenses_on_two_rings(self) -> None:
        array = (
            CameraSynthesizer()
            .add_ring(n=2, radius=2.0, lens=WEBCAM)
            .add_ring(n=2, radius=2.0, height=0.5, lens=MACHINE_VISION)
            .build()
        )
        for cam_id in [0, 1]:
            cam = array.cameras[cam_id]
            assert cam.matrix is not None
            assert cam.matrix[0, 0] == pytest.approx(WEBCAM.focal_px)
        for cam_id in [2, 3]:
            cam = array.cameras[cam_id]
            assert cam.matrix is not None
            assert cam.matrix[0, 0] == pytest.approx(MACHINE_VISION.focal_px)

    def test_resolution_from_profile(self) -> None:
        custom = LensProfile(
            focal_px=600.0,
            distortions=np.zeros(5, dtype=np.float64),
            resolution=(640, 480),
        )
        array = CameraSynthesizer().add_ring(n=2, radius=2.0, lens=custom).build()
        for cam in array.cameras.values():
            assert cam.size == (640, 480)
            assert cam.matrix is not None
            assert cam.matrix[0, 2] == pytest.approx(320.0)
            assert cam.matrix[1, 2] == pytest.approx(240.0)

    def test_distortions_copied_not_shared(self) -> None:
        array = CameraSynthesizer().add_ring(n=2, radius=2.0, lens=WEBCAM).build()
        cam0 = array.cameras[0]
        cam1 = array.cameras[1]
        assert cam0.distortions is not None
        assert cam1.distortions is not None
        cam0.distortions[0] = 999.0
        assert cam1.distortions[0] != 999.0

    def test_add_line_accepts_lens(self) -> None:
        array = CameraSynthesizer().add_line(n=3, spacing=1.0, lens=MACHINE_VISION).build()
        for cam in array.cameras.values():
            assert cam.matrix is not None
            assert cam.matrix[0, 0] == pytest.approx(MACHINE_VISION.focal_px)


class TestPerturbIntrinsics:
    def test_f_scale_uniform(self) -> None:
        array = CameraSynthesizer().add_ring(n=4, radius=2.0).build()
        perturbed = perturb_intrinsics(array, IntrinsicPerturbation(f_scale=1.03))

        for cam_id in array.cameras:
            orig = array.cameras[cam_id]
            pert = perturbed.cameras[cam_id]
            assert orig.matrix is not None and pert.matrix is not None
            assert pert.matrix[0, 0] == pytest.approx(orig.matrix[0, 0] * 1.03)
            assert pert.matrix[1, 1] == pytest.approx(orig.matrix[1, 1] * 1.03)
            # Principal point unchanged
            assert pert.matrix[0, 2] == pytest.approx(orig.matrix[0, 2])

    def test_k1_delta(self) -> None:
        array = CameraSynthesizer().add_ring(n=2, radius=2.0).build()
        perturbed = perturb_intrinsics(array, IntrinsicPerturbation(k1_delta=0.05))

        for cam_id in array.cameras:
            orig = array.cameras[cam_id]
            pert = perturbed.cameras[cam_id]
            assert orig.distortions is not None and pert.distortions is not None
            assert pert.distortions[0] == pytest.approx(orig.distortions[0] + 0.05)
            # Other distortion coefficients unchanged
            np.testing.assert_array_equal(pert.distortions[1:], orig.distortions[1:])

    def test_per_camera_dict(self) -> None:
        array = CameraSynthesizer().add_ring(n=4, radius=2.0).build()
        perturbation = {
            0: IntrinsicPerturbation(f_scale=1.05),
            2: IntrinsicPerturbation(f_scale=0.95),
        }
        perturbed = perturb_intrinsics(array, perturbation)

        orig_0 = array.cameras[0]
        pert_0 = perturbed.cameras[0]
        assert orig_0.matrix is not None and pert_0.matrix is not None
        assert pert_0.matrix[0, 0] == pytest.approx(orig_0.matrix[0, 0] * 1.05)

        # Camera 1 unperturbed
        orig_1 = array.cameras[1]
        pert_1 = perturbed.cameras[1]
        assert orig_1.matrix is not None and pert_1.matrix is not None
        np.testing.assert_array_equal(pert_1.matrix, orig_1.matrix)

    def test_does_not_mutate_input(self) -> None:
        array = CameraSynthesizer().add_ring(n=2, radius=2.0).build()
        orig_f = array.cameras[0].matrix[0, 0]  # type: ignore[index]
        perturb_intrinsics(array, IntrinsicPerturbation(f_scale=2.0))
        assert array.cameras[0].matrix[0, 0] == pytest.approx(orig_f)  # type: ignore[index]

    def test_identity_perturbation_is_copy(self) -> None:
        array = CameraSynthesizer().add_ring(n=2, radius=2.0).build()
        perturbed = perturb_intrinsics(array, IntrinsicPerturbation())

        for cam_id in array.cameras:
            orig = array.cameras[cam_id]
            pert = perturbed.cameras[cam_id]
            assert orig.matrix is not None and pert.matrix is not None
            np.testing.assert_array_equal(pert.matrix, orig.matrix)
            assert orig.distortions is not None and pert.distortions is not None
            np.testing.assert_array_equal(pert.distortions, orig.distortions)


if __name__ == "__main__":
    array = CameraSynthesizer().add_ring(n=4, radius=2.0).build()
    print(f"Default lens: f={array.cameras[0].matrix[0, 0]}")  # type: ignore[index]
    print(f"Default distortions: {array.cameras[0].distortions}")

    perturbed = perturb_intrinsics(array, IntrinsicPerturbation(f_scale=1.03, k1_delta=0.01))
    print(f"Perturbed f: {perturbed.cameras[0].matrix[0, 0]}")  # type: ignore[index]
    print(f"Perturbed k1: {perturbed.cameras[0].distortions[0]}")  # type: ignore[index]

    pytest.main([__file__, "-v"])
