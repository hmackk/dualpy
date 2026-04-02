import numpy as np
import pytest
from dualpy.core import DualArray

from tests.conftest import assert_dual_close, finite_diff_jvp

pytestmark = pytest.mark.unit


class TestDot:
    def test_1d(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.dot(a, b)
        assert isinstance(result, DualArray)
        expected_primal = np.dot([1.0, 2.0], [3.0, 4.0])
        expected_tangent = np.dot([0.1, 0.2], [3.0, 4.0]) + np.dot(
            [1.0, 2.0], [0.3, 0.4]
        )
        assert_dual_close(result, expected_primal, expected_tangent)

    def test_2d(self):
        a = DualArray(np.eye(2), np.array([[0.1, 0.2], [0.3, 0.4]]))
        b = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.zeros((2, 2)))
        result = np.dot(a, b)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array(
                [
                    [0.1 * 1 + 0.2 * 3, 0.1 * 2 + 0.2 * 4],
                    [0.3 * 1 + 0.4 * 3, 0.3 * 2 + 0.4 * 4],
                ]
            ),
        )

    def test_dual_with_plain_ndarray(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = np.array([3.0, 4.0])
        result = np.dot(a, b)
        assert isinstance(result, DualArray)
        expected_primal = np.dot([1.0, 2.0], [3.0, 4.0])
        expected_tangent = np.dot([0.1, 0.2], [3.0, 4.0])
        assert_dual_close(result, expected_primal, expected_tangent)

    def test_dot_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        v = np.array([0.1, -0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.dot(z, b), x, v)
        da = DualArray(x, v)
        result = np.dot(da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestInner:
    def test_1d(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.inner(a, b)
        assert isinstance(result, DualArray)
        expected_primal = np.inner([1.0, 2.0], [3.0, 4.0])
        expected_tangent = np.inner([0.1, 0.2], [3.0, 4.0]) + np.inner(
            [1.0, 2.0], [0.3, 0.4]
        )
        assert_dual_close(result, expected_primal, expected_tangent)

    def test_inner_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        v = np.array([0.1, -0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.inner(z, b), x, v)
        da = DualArray(x, v)
        result = np.inner(da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestOuter:
    def test_basic(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.0, 0.0]))
        result = np.outer(a, b)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.outer([1.0, 2.0], [3.0, 4.0]),
            np.outer([0.1, 0.2], [3.0, 4.0]),
        )

    def test_outer_vs_finite_diff(self):
        x = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        v = np.array([0.1, -0.2])
        fd = finite_diff_jvp(lambda z: np.outer(z, b), x, v)
        da = DualArray(x, v)
        result = np.outer(da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestTensordot:
    def test_2d(self):
        a = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2) * 0.1)
        b = DualArray(np.array([[5.0, 6.0], [7.0, 8.0]]), np.zeros((2, 2)))
        result = np.tensordot(a, b, axes=1)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(
            result.primal,
            np.tensordot([[1, 2], [3, 4]], [[5, 6], [7, 8]], axes=1),
            rtol=1e-7,
        )

    def test_tensordot_vs_finite_diff(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        v = np.eye(2) * 0.1
        fd = finite_diff_jvp(lambda z: np.tensordot(z, b, axes=1), x, v)
        da = DualArray(x, v)
        result = np.tensordot(da, b, axes=1)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestEinsum:
    def test_matrix_multiply(self):
        a = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2) * 0.1)
        b = DualArray(np.array([[5.0, 6.0], [7.0, 8.0]]), np.zeros((2, 2)))
        result = np.einsum("ij,jk->ik", a, b)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, a.primal @ b.primal, rtol=1e-7)

    def test_trace(self):
        a = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2) * 0.5)
        result = np.einsum("ii->", a)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array(5.0), np.array(1.0))

    def test_mixed_operands(self):
        a = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2) * 0.1)
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = np.einsum("ij,jk->ik", a, b)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, a.primal @ b, rtol=1e-7)
        expected_tangent = a.tangent @ b
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    def test_einsum_vs_finite_diff(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        v = np.eye(2) * 0.1
        fd = finite_diff_jvp(lambda z: np.einsum("ij,jk->ik", z, b), x, v)
        da = DualArray(x, v)
        result = np.einsum("ij,jk->ik", da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestTrace:
    def test_basic(self):
        a = DualArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        result = np.trace(a)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array(5.0), np.array(0.5))

    def test_trace_vs_finite_diff(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        fd = finite_diff_jvp(lambda z: np.trace(z), x, v)
        da = DualArray(x, v)
        result = np.trace(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestCross:
    def test_3d(self):
        a = DualArray(np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]))
        b = DualArray(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        result = np.cross(a, b)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.1]),
        )

    def test_dual_with_plain_ndarray(self):
        a = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        b = np.array([4.0, 5.0, 6.0])
        result = np.cross(a, b)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.cross(a.primal, b))
        expected_tangent = np.cross(a.tangent, b)
        np.testing.assert_allclose(result.tangent, expected_tangent)

    def test_cross_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        v = np.array([0.1, -0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.cross(z, b), x, v)
        da = DualArray(x, v)
        result = np.cross(da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestMatmulVecVec:
    def test_dot_product(self):
        x_p = np.array([1.0, 2.0, 3.0])
        x_t = np.array([0.1, 0.2, 0.3])
        y_p = np.array([4.0, 5.0, 6.0])
        y_t = np.array([0.4, 0.5, 0.6])
        result = DualArray(x_p, x_t) @ DualArray(y_p, y_t)
        expected_primal = x_p @ y_p
        expected_tangent = x_t @ y_p + x_p @ y_t
        assert_dual_close(result, expected_primal, expected_tangent)

    def test_matmul_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        v = np.array([0.1, -0.2, 0.3])
        fd = finite_diff_jvp(lambda z: z @ b, x, v)
        da = DualArray(x, v)
        result = da @ b
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestMatmulMatVec:
    def test_matrix_vector(self):
        A_p = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_t = np.array([[0.1, 0.2], [0.3, 0.4]])
        x_p = np.array([1.0, 1.0])
        x_t = np.array([1.0, 0.0])
        result = DualArray(A_p, A_t) @ DualArray(x_p, x_t)
        expected_primal = A_p @ x_p
        expected_tangent = A_t @ x_p + A_p @ x_t
        assert_dual_close(result, expected_primal, expected_tangent)


class TestMatmulMatMat:
    def test_matrix_matrix(self):
        A_p = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_t = np.array([[0.1, 0.2], [0.3, 0.4]])
        B_p = np.array([[5.0, 6.0], [7.0, 8.0]])
        B_t = np.array([[0.5, 0.6], [0.7, 0.8]])
        result = DualArray(A_p, A_t) @ DualArray(B_p, B_t)
        expected_primal = A_p @ B_p
        expected_tangent = A_t @ B_p + A_p @ B_t
        assert_dual_close(result, expected_primal, expected_tangent)


class TestVdot:
    def test_basic(self):
        a = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        b = DualArray(np.array([4.0, 5.0, 6.0]), np.array([0.4, 0.5, 0.6]))
        result = np.vdot(a, b)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.vdot(a.primal, b.primal))

    def test_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        v = np.array([0.1, -0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.vdot(z, b), x, v)
        da = DualArray(x, v)
        result = np.vdot(da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestKron:
    def test_basic(self):
        a = DualArray(np.array([[1.0, 0.0], [0.0, 1.0]]), np.eye(2) * 0.1)
        b = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.zeros((2, 2)))
        result = np.kron(a, b)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.kron(a.primal, b.primal))

    def test_vs_finite_diff(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        v = np.array([[0.1, -0.2], [0.3, -0.1]])
        fd = finite_diff_jvp(lambda z: np.kron(z, b), a, v)
        da = DualArray(a, v)
        result = np.kron(da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestNorm:
    def test_l2_vector(self):
        x = np.array([3.0, 4.0])
        t = np.array([0.1, 0.2])
        da = DualArray(x, t)
        result = np.linalg.norm(da)
        np.testing.assert_allclose(result.primal, 5.0)
        expected_t = (3.0 * 0.1 + 4.0 * 0.2) / 5.0
        np.testing.assert_allclose(result.tangent, expected_t, rtol=1e-7)

    def test_l2_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, -0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.linalg.norm(z), x, v)
        da = DualArray(x, v)
        result = np.linalg.norm(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_l1(self):
        x = np.array([3.0, -4.0])
        v = np.array([1.0, 1.0])
        fd = finite_diff_jvp(lambda z: np.linalg.norm(z, ord=1), x, v)
        da = DualArray(x, v)
        result = np.linalg.norm(da, ord=1)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_with_axis(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.linalg.norm(da, axis=1)
        np.testing.assert_allclose(result.primal, np.linalg.norm(x, axis=1))


class TestDet:
    def test_2x2(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        dA = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(A, dA)
        result = np.linalg.det(da)
        np.testing.assert_allclose(result.primal, np.linalg.det(A), rtol=1e-7)

    def test_vs_finite_diff(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        fd = finite_diff_jvp(lambda z: np.linalg.det(z), A, v)
        da = DualArray(A, v)
        result = np.linalg.det(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_3x3(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3))
        v = rng.standard_normal((3, 3))
        fd = finite_diff_jvp(lambda z: np.linalg.det(z), A, v)
        da = DualArray(A, v)
        result = np.linalg.det(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestInv:
    def test_2x2(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        dA = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(A, dA)
        result = np.linalg.inv(da)
        np.testing.assert_allclose(result.primal, np.linalg.inv(A), rtol=1e-7)

    def test_vs_finite_diff(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        fd = finite_diff_jvp(lambda z: np.linalg.inv(z), A, v)
        da = DualArray(A, v)
        result = np.linalg.inv(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_3x3(self):
        rng = np.random.default_rng(43)
        A = rng.standard_normal((3, 3)) + 3 * np.eye(3)
        v = rng.standard_normal((3, 3))
        fd = finite_diff_jvp(lambda z: np.linalg.inv(z), A, v)
        da = DualArray(A, v)
        result = np.linalg.inv(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestSolve:
    def test_basic(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        da = DualArray(A, np.zeros_like(A))
        db = DualArray(b, np.ones_like(b))
        result = np.linalg.solve(da, db)
        np.testing.assert_allclose(result.primal, np.linalg.solve(A, b), rtol=1e-7)

    def test_vs_finite_diff_b(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        v_b = np.array([0.1, -0.2])
        fd = finite_diff_jvp(lambda z: np.linalg.solve(A, z), b, v_b)
        da = DualArray(A, np.zeros_like(A))
        db = DualArray(b, v_b)
        result = np.linalg.solve(da, db)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_vs_finite_diff_A(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        v_A = np.array([[0.1, 0.2], [0.3, 0.4]])
        fd = finite_diff_jvp(lambda z: np.linalg.solve(z, b), A, v_A)
        da = DualArray(A, v_A)
        db = DualArray(b, np.zeros_like(b))
        result = np.linalg.solve(da, db)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestCholesky:
    def test_basic(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        da = DualArray(A, np.zeros_like(A))
        result = np.linalg.cholesky(da)
        np.testing.assert_allclose(result.primal, np.linalg.cholesky(A), rtol=1e-7)

    def test_vs_finite_diff(self):
        rng = np.random.default_rng(44)
        M = rng.standard_normal((3, 3))
        A = M @ M.T + 3 * np.eye(3)
        v = rng.standard_normal((3, 3))
        v = (v + v.T) / 2
        fd = finite_diff_jvp(lambda z: np.linalg.cholesky(z), A, v)
        da = DualArray(A, v)
        result = np.linalg.cholesky(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-4)


class TestEigh:
    def test_eigenvalues(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        dA = np.array([[0.1, 0.05], [0.05, 0.2]])
        da = DualArray(A, dA)
        w_result, V_result = np.linalg.eigh(da)
        w, V = np.linalg.eigh(A)
        np.testing.assert_allclose(w_result.primal, w, rtol=1e-7)
        assert isinstance(w_result, DualArray)
        assert isinstance(V_result, DualArray)

    def test_eigenvalues_vs_finite_diff(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        v = np.array([[0.1, 0.05], [0.05, 0.2]])
        fd = finite_diff_jvp(lambda z: np.linalg.eigh(z)[0], A, v)
        da = DualArray(A, v)
        w_result, _ = np.linalg.eigh(da)
        np.testing.assert_allclose(w_result.tangent, fd, rtol=1e-5)


class TestSvd:
    def test_singular_values(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        dA = np.zeros_like(A)
        da = DualArray(A, dA)
        U, s, Vt = np.linalg.svd(da, full_matrices=False)
        U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)
        np.testing.assert_allclose(s.primal, s_np, rtol=1e-7)
        assert isinstance(s, DualArray)

    def test_singular_values_vs_finite_diff(self):
        rng = np.random.default_rng(45)
        A = rng.standard_normal((3, 2))
        v = rng.standard_normal((3, 2))
        fd = finite_diff_jvp(lambda z: np.linalg.svd(z, full_matrices=False)[1], A, v)
        da = DualArray(A, v)
        _, s, _ = np.linalg.svd(da, full_matrices=False)
        np.testing.assert_allclose(s.tangent, fd, rtol=1e-4)


class TestQr:
    def test_primal(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        da = DualArray(A, np.zeros_like(A))
        Q, R = np.linalg.qr(da)
        Q_np, R_np = np.linalg.qr(A)
        np.testing.assert_allclose(Q.primal, Q_np, rtol=1e-7)
        np.testing.assert_allclose(R.primal, R_np, rtol=1e-7)
        assert isinstance(Q, DualArray)
        assert isinstance(R, DualArray)

    def test_vs_finite_diff_R(self):
        rng = np.random.default_rng(46)
        A = rng.standard_normal((3, 3)) + 2 * np.eye(3)
        v = rng.standard_normal((3, 3)) * 0.01
        fd = finite_diff_jvp(lambda z: np.linalg.qr(z)[1], A, v)
        da = DualArray(A, v)
        _, R = np.linalg.qr(da)
        np.testing.assert_allclose(R.tangent, fd, rtol=1e-3, atol=1e-12)


class TestLstsq:
    def test_basic(self):
        A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0, 2.0])
        da = DualArray(A, np.zeros_like(A))
        db = DualArray(b, np.ones_like(b))
        x_result, residuals, rank, sv = np.linalg.lstsq(da, db, rcond=None)
        x_np = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_allclose(x_result.primal, x_np, rtol=1e-7)
        assert isinstance(x_result, DualArray)

    def test_vs_finite_diff(self):
        A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0, 2.0])
        v_b = np.array([0.1, -0.2, 0.15])
        fd = finite_diff_jvp(lambda z: np.linalg.lstsq(A, z, rcond=None)[0], b, v_b)
        da = DualArray(A, np.zeros_like(A))
        db = DualArray(b, v_b)
        x_result, _, _, _ = np.linalg.lstsq(da, db, rcond=None)
        np.testing.assert_allclose(x_result.tangent, fd, rtol=1e-4)


class TestMatrixPower:
    def test_identity(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        dA = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(A, dA)
        result = np.linalg.matrix_power(da, 0)
        np.testing.assert_allclose(result.primal, np.eye(2))
        np.testing.assert_allclose(result.tangent, np.zeros((2, 2)))

    def test_power_1(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        dA = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(A, dA)
        result = np.linalg.matrix_power(da, 1)
        assert_dual_close(result, A, dA)

    def test_power_2(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        dA = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(A, dA)
        result = np.linalg.matrix_power(da, 2)
        np.testing.assert_allclose(result.primal, A @ A, rtol=1e-7)

    def test_vs_finite_diff(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        fd = finite_diff_jvp(lambda z: np.linalg.matrix_power(z, 3), A, v)
        da = DualArray(A, v)
        result = np.linalg.matrix_power(da, 3)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_negative_power(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        fd = finite_diff_jvp(lambda z: np.linalg.matrix_power(z, -2), A, v)
        da = DualArray(A, v)
        result = np.linalg.matrix_power(da, -2)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-4)


class TestMultiDot:
    def test_two_matrices(self):
        A = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2) * 0.1)
        B = DualArray(np.array([[5.0, 6.0], [7.0, 8.0]]), np.zeros((2, 2)))
        result = np.linalg.multi_dot([A, B])
        np.testing.assert_allclose(result.primal, A.primal @ B.primal, rtol=1e-7)

    def test_three_matrices(self):
        rng = np.random.default_rng(47)
        A = rng.standard_normal((2, 3))
        B = rng.standard_normal((3, 4))
        C = rng.standard_normal((4, 2))
        v = rng.standard_normal((2, 3))
        fd = finite_diff_jvp(lambda z: np.linalg.multi_dot([z, B, C]), A, v)
        da = DualArray(A, v)
        result = np.linalg.multi_dot([da, B, C])
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_vs_finite_diff_middle(self):
        rng = np.random.default_rng(48)
        A = rng.standard_normal((2, 3))
        B = rng.standard_normal((3, 4))
        C = rng.standard_normal((4, 2))
        v = rng.standard_normal((3, 4))
        fd = finite_diff_jvp(lambda z: np.linalg.multi_dot([A, z, C]), B, v)
        db = DualArray(B, v)
        result = np.linalg.multi_dot([A, db, C])
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)
