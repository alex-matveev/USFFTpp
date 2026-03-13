import unittest

import numpy as np

import usfftpp as us


def _reference_1d(points, f_in, n):
    """
    Простейшее дискретное неравномерное преобразование:
    F[k] = sum_j f_j * exp(-2*pi*i * (k - n/2) * x_j), x_j in [-0.5, 0.5].
    """
    k = np.arange(n) - n // 2
    # shape (n, M)
    phase = np.exp(-2j * np.pi * k[:, None] * points[None, :])
    return phase @ f_in


def _reference_2d(points, f_in, n0, n1):
    """
    Простейшее 2D-преобразование:
    F[k0,k1] = sum_j f_j * exp(-2*pi*i * (k0 - n0/2)*x_j + (k1 - n1/2)*y_j).
    """
    k0 = np.arange(n0) - n0 // 2
    k1 = np.arange(n1) - n1 // 2
    x = points[:, 0]
    y = points[:, 1]

    phase0 = np.exp(-2j * np.pi * k0[:, None] * x[None, :])  # (n0, M)
    phase1 = np.exp(-2j * np.pi * k1[:, None] * y[None, :])  # (n1, M)

    # F[k0,k1] = sum_j f_j * exp(...) = sum_j f_j * phase0[k0,j] * phase1[k1,j]
    F = np.zeros((n0, n1), dtype=f_in.dtype)
    for j in range(points.shape[0]):
        F += f_in[j] * np.outer(phase0[:, j], phase1[:, j])
    return F


class TestUSFFTppAgainstReference(unittest.TestCase):
    def test_plan1d_float_forward_matches_reference(self):
        n = 16
        rng = np.random.default_rng(0)
        points = (rng.random(8).astype(np.float32) - 0.5) * np.float32(0.98)
        eps = 1e-3

        plan = us.Plan1d(n, points, eps)

        f_in = (rng.standard_normal(points.shape[0]) + 1j * rng.standard_normal(points.shape[0])).astype(
            np.complex64
        )
        f_out = np.zeros(n, dtype=np.complex64)

        plan.nonuniform_to_uniform(f_in, f_out, us.FourierDirection.forward)

        ref = _reference_1d(points.astype(np.float32), f_in.astype(np.complex128), n).astype(np.complex64)

        np.testing.assert_allclose(f_out, ref, rtol=1e-2, atol=1e-3)

    def test_plan1d_double_forward_matches_reference(self):
        n = 16
        rng = np.random.default_rng(1)
        points = (rng.random(8).astype(np.float64) - 0.5) * 0.98
        eps = 1e-6

        plan = us.Plan1d(n, points, eps)

        f_in = rng.standard_normal(points.shape[0]) + 1j * rng.standard_normal(points.shape[0])
        f_in = f_in.astype(np.complex128)
        f_out = np.zeros(n, dtype=np.complex128)

        plan.nonuniform_to_uniform(f_in, f_out, us.FourierDirection.forward)

        ref = _reference_1d(points, f_in, n)

        np.testing.assert_allclose(f_out, ref, rtol=1e-5, atol=1e-7)

    def test_plan2d_float_forward_matches_reference(self):
        n0, n1 = 8, 8
        rng = np.random.default_rng(2)
        points = (rng.random((4, 2)).astype(np.float32) - 0.5).astype(np.float32)
        eps = 1e-3

        plan = us.Plan2d(n0, n1, points, eps)

        f_in = (rng.standard_normal(points.shape[0]) + 1j * rng.standard_normal(points.shape[0])).astype(
            np.complex64
        )
        f_out = np.zeros((n0, n1), dtype=np.complex64)

        plan.nonuniform_to_uniform(f_in, f_out, us.FourierDirection.forward)

        ref = _reference_2d(points.astype(np.float32), f_in.astype(np.complex128), n0, n1).astype(
            np.complex64
        )

        np.testing.assert_allclose(f_out, ref, rtol=1e-2, atol=1e-3)

    def test_plan2d_double_forward_matches_reference(self):
        n0, n1 = 8, 8
        rng = np.random.default_rng(3)
        points = (rng.random((4, 2)) - 0.5).astype(np.float64)
        eps = 1e-6

        plan = us.Plan2d(n0, n1, points, eps)

        f_in = (rng.standard_normal(points.shape[0]) + 1j * rng.standard_normal(points.shape[0])).astype(
            np.complex128
        )
        f_out = np.zeros((n0, n1), dtype=np.complex128)

        plan.nonuniform_to_uniform(f_in, f_out, us.FourierDirection.forward)

        ref = _reference_2d(points, f_in, n0, n1)

        np.testing.assert_allclose(f_out, ref, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()


