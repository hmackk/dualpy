import numpy as np
import pytest
from dualpy.differentiation import derivative, gradient, hessian, jacobian, jvp

from tests.conftest import finite_diff_jvp

pytestmark = pytest.mark.functional


class TestRosenbrock:
    @staticmethod
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def test_gradient(self):
        x = np.array([1.0, 1.0])
        grad = gradient(self.rosenbrock)(x)
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-10)

    def test_gradient_away_from_minimum(self):
        x = np.array([0.0, 0.0])
        grad = gradient(self.rosenbrock)(x)
        expected = np.array([-2.0, 0.0])
        np.testing.assert_allclose(grad, expected, atol=1e-10)

    def test_hessian_symmetry(self):
        x = np.array([1.5, 2.0])
        H = hessian(self.rosenbrock)(x)
        np.testing.assert_allclose(H, H.T, atol=1e-10)

    def test_hessian_at_minimum(self):
        x = np.array([1.0, 1.0])
        H = hessian(self.rosenbrock)(x)
        expected = np.array([[802.0, -400.0], [-400.0, 200.0]])
        np.testing.assert_allclose(H, expected, atol=1e-8)


class TestSoftmax:
    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def test_jacobian_rows_sum_to_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        J = jacobian(self.softmax)(x)
        row_sums = np.sum(J, axis=1)
        np.testing.assert_allclose(row_sums, np.zeros(3), atol=1e-10)

    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        primal, tangent = jvp(self.softmax, x, np.ones(3))
        np.testing.assert_allclose(np.sum(primal), 1.0, atol=1e-12)
        np.testing.assert_allclose(tangent.sum(), 0.0, atol=1e-10)


class TestLinearRegression:
    def test_mse_gradient(self):
        np.random.seed(42)
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = np.array([1.0, 2.0, 3.0])
        x = np.array([0.5, 0.5])

        def f(x):
            return np.sum((A @ x - b) ** 2) / len(b)

        grad = gradient(f)(x)
        expected = 2 * A.T @ (A @ x - b) / len(b)
        np.testing.assert_allclose(grad, expected, rtol=1e-7)


class TestLogisticLoss:
    def test_derivative(self):
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def loss(x):
            s = sigmoid(x)
            return -np.log(s)

        x = np.array(1.0)
        d = derivative(loss)(x)
        fd = finite_diff_jvp(loss, x, np.array(1.0))
        np.testing.assert_allclose(d, fd, rtol=1e-5)


class TestNewtonStep:
    def test_converges_for_sqrt(self):
        def f(x):
            return x**2 - 2.0

        df = derivative(f)
        x = np.array(1.0)
        for _ in range(10):
            x = x - f(x) / df(x)
        np.testing.assert_allclose(x, np.sqrt(2.0), rtol=1e-10)


class TestPhysics:
    def test_harmonic_force(self):
        k = 2.0

        def energy(x):
            return 0.5 * k * np.sum(x**2)

        x = np.array([1.0, 2.0, 3.0])
        force = -gradient(energy)(x)
        expected = -k * x
        np.testing.assert_allclose(force, expected, atol=1e-10)


class TestChainOfOperations:
    def test_exp_sin_matmul_gradient(self):
        A = np.array([[1.0, 0.5], [0.5, 1.0]])

        def f(x):
            return np.sum(np.exp(np.sin(x @ A)))

        x = np.array([0.5, 0.3])
        grad = gradient(f)(x)
        fd = np.array(
            [
                finite_diff_jvp(f, x, np.array([1.0, 0.0])),
                finite_diff_jvp(f, x, np.array([0.0, 1.0])),
            ]
        )
        np.testing.assert_allclose(grad, fd, rtol=1e-5)
