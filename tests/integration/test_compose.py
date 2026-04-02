import numpy as np
import pytest
from dualpy.core import DualArray
from dualpy.differentiation import derivative, gradient, hessian, hvp, jacobian, jvp

from tests.conftest import finite_diff_jvp

pytestmark = pytest.mark.integration


class TestJacobianOfReductions:
    def test_jacobian_of_sum(self):
        x = np.array([1.0, 2.0, 3.0])
        result = jacobian(lambda z: np.sum(z**2))(x)
        np.testing.assert_allclose(result, 2 * x, rtol=1e-7)

    def test_jacobian_of_prod(self):
        x = np.array([2.0, 3.0, 4.0])
        J = jacobian(lambda z: np.prod(z))(x)
        fd = np.array([finite_diff_jvp(np.prod, x, e) for e in np.eye(len(x))])
        np.testing.assert_allclose(J, fd, rtol=1e-5)

    def test_jacobian_of_max(self):
        x = np.array([1.0, 5.0, 3.0])
        J = gradient(lambda z: np.max(z))(x)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(J, expected, atol=1e-12)

    def test_jacobian_of_sort(self):
        x = np.array([3.0, 1.0, 2.0])
        J = jacobian(lambda z: np.sort(z))(x)
        expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(J, expected, atol=1e-12)


class TestComposePipelines:
    def test_split_arith_concat_roundtrip(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, v)
        parts = np.split(da, 2)
        combined = parts[0] + parts[1]
        result = np.concatenate([combined, combined])
        assert isinstance(result, DualArray)
        assert result.shape == (4,)

    def test_jvp_with_reduction_inside(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, -0.2, 0.3])
        _, tangent = jvp(lambda z: np.sum(np.sin(z)), x, v)
        fd = finite_diff_jvp(lambda z: np.sum(np.sin(z)), x, v)
        np.testing.assert_allclose(tangent, fd, rtol=1e-5)

    def test_hvp_with_reduction(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, -0.2, 0.3])

        def f(z):
            return np.sum(z**3)

        hvp_result = hvp(f, v)(x)
        H = hessian(f)(x)
        expected = H @ v
        np.testing.assert_allclose(hvp_result, expected, atol=1e-10)

    def test_nested_derivative_through_exp_log(self):
        def f(x):
            return np.exp(np.log(x))

        df = derivative(f)
        np.testing.assert_allclose(df(2.0), 1.0, atol=1e-10)

    def test_gradient_through_matmul(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])

        def f(x):
            return x @ A @ x

        x = np.array([1.0, 2.0])
        grad = gradient(f)(x)
        expected = (A + A.T) @ x
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_jacobian_of_where(self):
        x = np.array([1.0, -2.0, 3.0])

        def f(z):
            return np.where(z > 0, z**2, -z)

        J = jacobian(f)(x)
        for i in range(len(x)):
            e = np.zeros(len(x))
            e[i] = 1.0
            fd_col = finite_diff_jvp(f, x, e)
            np.testing.assert_allclose(J[:, i], fd_col, rtol=1e-4)


class TestNewUfuncCompositions:
    def test_logaddexp_in_logsumexp(self):
        def logsumexp(z):
            result = z[0]
            for i in range(1, len(z)):
                result = np.logaddexp(result, z[i])
            return result

        x = np.array([1.0, 2.0, 3.0])
        grad = gradient(logsumexp)(x)
        expected = np.exp(x) / np.sum(np.exp(x))
        np.testing.assert_allclose(grad, expected, rtol=1e-6)

    def test_hypot_gradient(self):
        def f(z):
            return np.sum(np.hypot(z, np.ones_like(z)))

        x = np.array([3.0, 4.0])
        grad = gradient(f)(x)
        expected = x / np.hypot(x, np.ones_like(x))
        np.testing.assert_allclose(grad, expected, rtol=1e-6)

    def test_gradient_through_norm(self):
        def f(z):
            return np.linalg.norm(z)

        x = np.array([3.0, 4.0])
        grad = gradient(f)(x)
        expected = x / np.linalg.norm(x)
        np.testing.assert_allclose(grad, expected, rtol=1e-7)

    def test_gradient_through_det(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])

        def f(z):
            return np.linalg.det(z)

        J = jacobian(f)(A)
        for i in range(2):
            for j in range(2):
                e = np.zeros_like(A)
                e[i, j] = 1.0
                fd = finite_diff_jvp(f, A, e)
                np.testing.assert_allclose(J[i, j], fd, rtol=1e-5)

    def test_gradient_through_solve(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])

        def f(z):
            return np.sum(np.linalg.solve(A, z) ** 2)

        grad = gradient(f)(b)
        fd_grad = np.array([finite_diff_jvp(f, b, e) for e in np.eye(len(b))])
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)

    def test_gradient_through_inv(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])

        def f(z):
            return np.sum(np.linalg.inv(z))

        J = jacobian(f)(A)
        for i in range(2):
            for j in range(2):
                e = np.zeros_like(A)
                e[i, j] = 1.0
                fd = finite_diff_jvp(f, A, e)
                np.testing.assert_allclose(J[i, j], fd, rtol=1e-5)

    def test_diff_then_sum(self):
        def f(z):
            return np.sum(np.diff(z) ** 2)

        x = np.array([1.0, 3.0, 2.0, 5.0])
        grad = gradient(f)(x)
        fd_grad = np.array([finite_diff_jvp(f, x, e) for e in np.eye(len(x))])
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)

    def test_convolve_gradient(self):
        kernel = np.array([1.0, -1.0])

        def f(z):
            return np.sum(np.convolve(z, kernel, mode="valid") ** 2)

        x = np.array([1.0, 3.0, 2.0, 5.0])
        grad = gradient(f)(x)
        fd_grad = np.array([finite_diff_jvp(f, x, e) for e in np.eye(len(x))])
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)

    def test_clip_with_floor_ceil(self):
        x = np.array([0.5, 1.5, 2.5, 3.5])
        da = DualArray(x, np.ones_like(x))
        floored = np.floor(da)
        ceiled = np.ceil(da)
        assert isinstance(floored, DualArray)
        assert isinstance(ceiled, DualArray)
        np.testing.assert_allclose(floored.tangent, np.zeros_like(x))
        np.testing.assert_allclose(ceiled.tangent, np.zeros_like(x))

    def test_isfinite_as_mask(self):
        x = np.array([1.0, np.inf, 2.0, np.nan])
        da = DualArray(x, np.array([0.1, 0.2, 0.3, 0.4]))
        mask = np.isfinite(da)
        np.testing.assert_array_equal(mask, np.array([True, False, True, False]))
