import numpy as np
import pytest
from dualpy.core import DualArray

from tests.conftest import assert_dual_close

pytestmark = pytest.mark.unit


class TestNpArray:
    @pytest.mark.xfail(
        reason=(
            "np.array does not dispatch through __array_function__; "
            "DualArray needs __array__ or a different construction path"
        ),
    )
    def test_np_array_of_dual_arrays(self):
        a = DualArray(np.array(1.0), np.array(0.1))
        b = DualArray(np.array(2.0), np.array(0.2))
        result = np.array([a, b])
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([1.0, 2.0]))
        np.testing.assert_allclose(result.tangent, np.array([0.1, 0.2]))


class TestZerosLike:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.zeros_like(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([0.0, 0.0]), np.array([0.0, 0.0]))


class TestOnesLike:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.ones_like(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([1.0, 1.0]), np.array([0.0, 0.0]))


class TestFullLike:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.full_like(da, 7.0)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([7.0, 7.0]), np.array([0.0, 0.0]))


class TestCopy:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.copy(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([1.0, 2.0]), np.array([0.1, 0.2]))


class TestConstructionIntInput:
    def test_zeros_like_int_dual(self):
        da = DualArray(np.array([1, 2, 3]))
        result = np.zeros_like(da)
        assert isinstance(result, DualArray)


class TestEmptyLike:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.empty_like(da)
        assert isinstance(result, DualArray)
        assert result.primal.shape == (2,)
        assert result.tangent.shape == (2,)


class TestDiag:
    def test_1d_to_2d(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.diag(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.diag([1.0, 2.0, 3.0]), np.diag([0.1, 0.2, 0.3]))

    def test_2d_to_1d(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(p, t)
        result = np.diag(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([1.0, 4.0]), np.array([0.1, 0.4]))

    def test_with_offset(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.diag(da, k=1)
        assert isinstance(result, DualArray)
        assert result.primal.shape == (3, 3)


class TestDiagonal:
    def test_basic(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(p, t)
        result = np.diagonal(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([1.0, 4.0]), np.array([0.1, 0.4]))

    def test_with_offset(self):
        p = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        t = np.ones((3, 3)) * 0.1
        da = DualArray(p, t)
        result = np.diagonal(da, offset=1)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([2.0, 6.0]))


class TestMeshgrid:
    def test_basic(self):
        x = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        y = DualArray(np.array([3.0, 4.0, 5.0]), np.array([0.3, 0.4, 0.5]))
        result = np.meshgrid(x, y)
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, DualArray)
        xg, yg = np.meshgrid(x.primal, y.primal)
        np.testing.assert_allclose(result[0].primal, xg)
        np.testing.assert_allclose(result[1].primal, yg)


class TestTriu:
    def test_basic(self):
        p = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        t = np.ones((3, 3)) * 0.1
        da = DualArray(p, t)
        result = np.triu(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.triu(p), np.triu(t))

    def test_with_k(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(p, t)
        result = np.triu(da, k=1)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.triu(p, k=1), np.triu(t, k=1))


class TestTril:
    def test_basic(self):
        p = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        t = np.ones((3, 3)) * 0.1
        da = DualArray(p, t)
        result = np.tril(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.tril(p), np.tril(t))

    def test_with_k(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(p, t)
        result = np.tril(da, k=-1)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.tril(p, k=-1), np.tril(t, k=-1))
