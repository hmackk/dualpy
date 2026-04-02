import numpy as np
import pytest
from dualpy.differentiation import (
    derivative,
    gradient,
    hessian,
    hvp,
    jacobian,
    jvp,
    nth_derivative,
)

from tests.conftest import finite_diff_jvp

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# jvp with tuple primals / tangents
# ---------------------------------------------------------------------------


class TestJvpMultiArg:
    def test_two_scalar_args(self):
        def f(x, y):
            return x**2 * y

        p, t = jvp(f, (np.array(3.0), np.array(2.0)), (np.array(1.0), np.array(0.0)))
        np.testing.assert_allclose(p, 18.0, atol=1e-12)
        np.testing.assert_allclose(t, 12.0, atol=1e-12)

    def test_two_vector_args(self):
        def f(x, y):
            return np.sum(x * y)

        x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        dx, dy = np.array([1.0, 0.0]), np.array([0.0, 0.0])
        p, t = jvp(f, (x, y), (dx, dy))
        np.testing.assert_allclose(p, 11.0, atol=1e-12)
        np.testing.assert_allclose(t, 3.0, atol=1e-12)

    def test_three_args(self):
        def f(a, b, c):
            return a * b + c

        p, t = jvp(
            f,
            (np.array(2.0), np.array(3.0), np.array(5.0)),
            (np.array(1.0), np.array(0.0), np.array(0.0)),
        )
        np.testing.assert_allclose(p, 11.0, atol=1e-12)
        np.testing.assert_allclose(t, 3.0, atol=1e-12)

    def test_shape_mismatch_in_tuple(self):
        def f(x, y):
            return x + y

        with pytest.raises(ValueError, match="shape mismatch"):
            jvp(
                f,
                (np.array([1.0, 2.0]), np.array(3.0)),
                (np.array([1.0]), np.array(1.0)),
            )

    def test_matches_finite_differences(self):
        def f(x, y):
            return np.stack([x[0] * y, x[1] ** 2 + y])

        x = np.array([2.0, 3.0])
        y = np.array(4.0)
        dx = np.array([0.3, -0.5])
        dy = np.array(0.7)
        _, tangent = jvp(f, (x, y), (dx, dy))

        def fd_fn(z):
            return np.array([z[0] * z[2], z[1] ** 2 + z[2]])

        fd = finite_diff_jvp(
            fd_fn,
            np.array([*x, y]),
            np.array([*dx, dy]),
        )
        np.testing.assert_allclose(tangent, fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# jacobian with argnums
# ---------------------------------------------------------------------------


class TestJacobianArgnums:
    def test_argnums_0(self):
        def f(x, y):
            return x**2 * y

        J = jacobian(f, argnums=0)(np.array(3.0), np.array(2.0))
        np.testing.assert_allclose(J, 12.0, atol=1e-12)

    def test_argnums_1(self):
        def f(x, y):
            return x**2 * y

        J = jacobian(f, argnums=1)(np.array(3.0), np.array(2.0))
        np.testing.assert_allclose(J, 9.0, atol=1e-12)

    def test_argnums_tuple(self):
        def f(x, y):
            return x**2 * y

        J0, J1 = jacobian(f, argnums=(0, 1))(np.array(3.0), np.array(2.0))
        np.testing.assert_allclose(J0, 12.0, atol=1e-12)
        np.testing.assert_allclose(J1, 9.0, atol=1e-12)

    def test_vector_and_scalar(self):
        def f(x, y):
            return np.stack([x[0] * y, x[1] * y])

        x = np.array([2.0, 3.0])
        y = np.array(4.0)
        J0 = jacobian(f, argnums=0)(x, y)
        expected_J0 = np.array([[4.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(J0, expected_J0, atol=1e-12)
        J1 = jacobian(f, argnums=1)(x, y)
        expected_J1 = np.array([2.0, 3.0])
        np.testing.assert_allclose(J1, expected_J1, atol=1e-12)

    def test_kwargs_passthrough(self):
        def f(x, y, scale=1.0):
            return x**2 * y * scale

        J = jacobian(f, argnums=0)(np.array(3.0), np.array(2.0), scale=0.5)
        np.testing.assert_allclose(J, 6.0, atol=1e-12)

    def test_finite_diff_cross_check(self):
        def f(x, y):
            return np.stack([np.sin(x[0]) * y[0], np.exp(x[1] + y[1])])

        x = np.array([1.0, 2.0])
        y = np.array([0.5, -1.0])
        J0 = jacobian(f, argnums=0)(x, y)
        for i in range(2):
            e = np.zeros(2)
            e[i] = 1.0

            def fd_slice(z):
                return f(z, y)

            fd = finite_diff_jvp(fd_slice, x, e)
            np.testing.assert_allclose(J0[:, i], fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# gradient with argnums
# ---------------------------------------------------------------------------


class TestGradientArgnums:
    def test_argnums_0(self):
        def f(x, y):
            return np.sum(x**2) + y**2

        g = gradient(f, argnums=0)(np.array([3.0, 4.0]), np.array(1.0))
        np.testing.assert_allclose(g, np.array([6.0, 8.0]), atol=1e-12)

    def test_argnums_1_scalar(self):
        def f(x, y):
            return np.sum(x**2) + y**2

        g = gradient(f, argnums=1)(np.array([3.0, 4.0]), np.array(1.0))
        np.testing.assert_allclose(g, 2.0, atol=1e-12)

    def test_argnums_tuple(self):
        def f(x, y):
            return np.sum(x**2) + y**2

        g0, g1 = gradient(f, argnums=(0, 1))(np.array([3.0, 4.0]), np.array(1.0))
        np.testing.assert_allclose(g0, np.array([6.0, 8.0]), atol=1e-12)
        np.testing.assert_allclose(g1, 2.0, atol=1e-12)

    def test_rejects_vector_valued(self):
        def f(x, y):
            return np.stack([x[0], y])

        with pytest.raises(ValueError, match="scalar-valued"):
            gradient(f, argnums=0)(np.array([1.0, 2.0]), np.array(3.0))


# ---------------------------------------------------------------------------
# derivative with argnums
# ---------------------------------------------------------------------------


class TestDerivativeArgnums:
    def test_argnums_0(self):
        def f(x, y):
            return x**3 * y

        d = derivative(f, argnums=0)(2.0, 3.0)
        np.testing.assert_allclose(d, 36.0, atol=1e-12)

    def test_argnums_1(self):
        def f(x, y):
            return x**3 * y

        d = derivative(f, argnums=1)(2.0, 3.0)
        np.testing.assert_allclose(d, 8.0, atol=1e-12)

    def test_argnums_tuple(self):
        def f(x, y):
            return x**3 * y

        d0, d1 = derivative(f, argnums=(0, 1))(2.0, 3.0)
        np.testing.assert_allclose(d0, 36.0, atol=1e-12)
        np.testing.assert_allclose(d1, 8.0, atol=1e-12)

    def test_rejects_vector_input(self):
        def f(x, y):
            return np.sum(x) * y

        with pytest.raises(ValueError, match="scalar input"):
            derivative(f, argnums=0)(np.array([1.0, 2.0]), 3.0)


# ---------------------------------------------------------------------------
# hessian with argnums
# ---------------------------------------------------------------------------


class TestHessianArgnums:
    def test_same_arg(self):
        def f(x, y):
            return np.sum(x**3) * y

        H = hessian(f, argnums=0)(np.array([1.0, 2.0]), 3.0)
        expected = np.diag([6.0, 12.0]) * 3.0
        np.testing.assert_allclose(H, expected, atol=1e-10)

    def test_mixed_partial(self):
        def f(x, y):
            return np.sum(x**2) * y

        H_01 = hessian(f, argnums=(0, 1))(np.array([1.0, 2.0]), 3.0)
        np.testing.assert_allclose(H_01, np.array([2.0, 4.0]), atol=1e-10)

    def test_mixed_partial_symmetry(self):
        def f(x, y):
            return x[0] * y[0] + x[1] * y[1]

        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        H_01 = hessian(f, argnums=(0, 1))(x, y)
        H_10 = hessian(f, argnums=(1, 0))(x, y)
        np.testing.assert_allclose(H_01, H_10.T, atol=1e-12)

    def test_invalid_tuple_length(self):
        def f(x, y, z):
            return x * y * z

        with pytest.raises(ValueError, match="pair"):
            hessian(f, argnums=(0, 1, 2))


# ---------------------------------------------------------------------------
# hvp with argnums
# ---------------------------------------------------------------------------


class TestHvpArgnums:
    def test_matches_full_hessian(self):
        def f(x, y):
            return np.sum(x**3) + y * np.sum(x)

        x = np.array([1.0, 2.0])
        y = np.array(3.0)
        v = np.array([0.5, -1.0])
        hvp_result = hvp(f, v, argnums=0)(x, y)
        H = hessian(f, argnums=0)(x, y)
        np.testing.assert_allclose(hvp_result, H @ v, atol=1e-10)


# ---------------------------------------------------------------------------
# nth_derivative with argnums
# ---------------------------------------------------------------------------


class TestNthDerivativeArgnums:
    def test_second_partial(self):
        def f(x, y):
            return x**4 * y

        d2 = nth_derivative(f, 2, argnums=0)(1.0, 2.0)
        np.testing.assert_allclose(d2, 24.0, atol=1e-10)

    def test_third_partial(self):
        def f(x, y):
            return x**4 * y

        d3 = nth_derivative(f, 3, argnums=0)(1.0, 2.0)
        np.testing.assert_allclose(d3, 48.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Backward compatibility — all existing single-arg patterns must work
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_jacobian_single_arg(self):
        def f(x):
            return x**3

        J = jacobian(f)(np.array(2.0))
        np.testing.assert_allclose(J, 12.0, atol=1e-12)

    def test_jacobian_with_v(self):
        def f(x):
            return x**2

        result = jacobian(f, v=np.array(1.0))(np.array(3.0))
        np.testing.assert_allclose(result, 6.0, atol=1e-12)

    def test_gradient_single_arg(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        g = gradient(f)(np.array([3.0, 4.0]))
        np.testing.assert_allclose(g, np.array([6.0, 8.0]), atol=1e-12)

    def test_gradient_with_v(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        d = gradient(f, v=np.array([1.0, 0.0]))(np.array([3.0, 4.0]))
        np.testing.assert_allclose(d, 6.0, atol=1e-12)

    def test_derivative_single_arg(self):
        def f(x):
            return x**2

        np.testing.assert_allclose(derivative(f)(2.0), 4.0, atol=1e-12)

    def test_hessian_single_arg(self):
        def f(x):
            return x[0] ** 2 + 3 * x[1] ** 2

        H = hessian(f)(np.array([1.0, 1.0]))
        expected = np.array([[2.0, 0.0], [0.0, 6.0]])
        np.testing.assert_allclose(H, expected, atol=1e-12)

    def test_hvp_single_arg(self):
        def f(x):
            return x[0] ** 2 + 3 * x[1] ** 2

        result = hvp(f, np.array([1.0, 0.0]))(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, np.array([2.0, 0.0]), atol=1e-12)

    def test_jvp_single_arg(self):
        def f(x):
            return x**2

        p, t = jvp(f, np.array(3.0), np.array(1.0))
        np.testing.assert_allclose(p, 9.0, atol=1e-12)
        np.testing.assert_allclose(t, 6.0, atol=1e-12)

    def test_nth_derivative_single_arg(self):
        def f(x):
            return x**4

        np.testing.assert_allclose(nth_derivative(f, 3)(1.0), 24.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_argnums_out_of_range(self):
        def f(x, y):
            return x + y

        with pytest.raises(IndexError):
            jacobian(f, argnums=5)(np.array(1.0), np.array(2.0))

    def test_different_shapes(self):
        def f(x, y):
            return np.sum(x) * np.sum(y**2)

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0])
        g0 = gradient(f, argnums=0)(x, y)
        assert g0.shape == (3,)
        np.testing.assert_allclose(g0, np.sum(y**2) * np.ones(3), atol=1e-12)
        g1 = gradient(f, argnums=1)(x, y)
        assert g1.shape == (2,)
        np.testing.assert_allclose(g1, np.sum(x) * 2 * y, atol=1e-12)

    def test_nested_differentiation_with_argnums(self):
        def f(x, y):
            return x**3 * y

        d2 = derivative(derivative(f, argnums=0), argnums=0)(2.0, 3.0)
        np.testing.assert_allclose(d2, 36.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Practical multi-arg scenarios
# ---------------------------------------------------------------------------


class TestRealWorldMultiArg:
    def test_mse_gradient_wrt_weights(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = np.array([1.0, 2.0, 3.0])

        def mse(w, data, targets):
            residuals = data @ w - targets
            return np.sum(residuals**2) / len(targets)

        x = np.array([0.5, 0.5])
        grad_w = gradient(mse, argnums=0)(x, A, b)
        expected = 2 * A.T @ (A @ x - b) / len(b)
        np.testing.assert_allclose(grad_w, expected, rtol=1e-7)

    def test_bilinear_form(self):
        def bilinear(x, A, y):
            return x @ A @ y

        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        gx = gradient(bilinear, argnums=0)(x, A, y)
        np.testing.assert_allclose(gx, A @ y, atol=1e-12)
        gy = gradient(bilinear, argnums=2)(x, A, y)
        np.testing.assert_allclose(gy, A.T @ x, atol=1e-12)

    def test_weighted_norm(self):
        def weighted_norm(x, w):
            return np.sum(w * x**2)

        x = np.array([1.0, 2.0, 3.0])
        w = np.array([0.5, 1.0, 2.0])
        gx, gw = gradient(weighted_norm, argnums=(0, 1))(x, w)
        np.testing.assert_allclose(gx, 2 * w * x, atol=1e-12)
        np.testing.assert_allclose(gw, x**2, atol=1e-12)
