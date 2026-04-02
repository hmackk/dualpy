import numpy as np
import pytest
from dualpy.core import DualArray

from tests.conftest import assert_dual_close, finite_diff_jvp

pytestmark = pytest.mark.unit


class TestWhere:
    def test_basic(self):
        cond = np.array([True, False, True])
        x = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        y = DualArray(np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0]))
        result = np.where(cond, x, y)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 20.0, 3.0]),
            np.array([0.1, 2.0, 0.3]),
        )

    def test_scalar_branches(self):
        cond = np.array([True, False])
        x = DualArray(np.array([5.0, 6.0]), np.array([0.5, 0.6]))
        result = np.where(cond, x, 0.0)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([5.0, 0.0]),
            np.array([0.5, 0.0]),
        )


class TestClip:
    def test_basic(self):
        da = DualArray(np.array([0.5, 1.5, 2.5]), np.array([1.0, 1.0, 1.0]))
        result = np.clip(da, 1.0, 2.0)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 1.5, 2.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    def test_no_clip(self):
        da = DualArray(np.array([1.5]), np.array([1.0]))
        result = np.clip(da, 0.0, 3.0)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([1.5]), np.array([1.0]))

    def test_clip_vs_finite_diff(self):
        x = np.array([0.5, 1.5, 2.5])
        v = np.array([0.1, 0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.clip(z, 1.0, 2.0), x, v)
        da = DualArray(x, v)
        result = np.clip(da, 1.0, 2.0)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestSort:
    def test_1d(self):
        da = DualArray(np.array([3.0, 1.0, 2.0]), np.array([0.3, 0.1, 0.2]))
        result = np.sort(da)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.2, 0.3]),
        )

    def test_2d_axis0(self):
        da = DualArray(
            np.array([[3.0, 1.0], [1.0, 3.0]]),
            np.array([[0.3, 0.1], [0.1, 0.3]]),
        )
        result = np.sort(da, axis=0)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[1.0, 1.0], [3.0, 3.0]]),
            np.array([[0.1, 0.1], [0.3, 0.3]]),
        )


class TestArgsort:
    def test_basic(self):
        da = DualArray(np.array([3.0, 1.0, 2.0]), np.array([0.3, 0.1, 0.2]))
        result = np.argsort(da)
        np.testing.assert_array_equal(result, np.argsort(da.primal))
        assert not isinstance(result, DualArray)

    def test_with_axis(self):
        p = np.array([[3.0, 1.0], [2.0, 4.0]])
        t = np.ones_like(p) * 0.1
        da = DualArray(p, t)
        result = np.argsort(da, axis=0)
        np.testing.assert_array_equal(result, np.argsort(p, axis=0))


class TestSearchsorted:
    def test_basic(self):
        da = DualArray(np.array([1.0, 3.0, 5.0, 7.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        result = np.searchsorted(da, 4.0)
        assert result == np.searchsorted(da.primal, 4.0)

    def test_array_values(self):
        da = DualArray(np.array([1.0, 3.0, 5.0]), np.array([0.1, 0.2, 0.3]))
        result = np.searchsorted(da, [2.0, 4.0])
        np.testing.assert_array_equal(result, np.searchsorted(da.primal, [2.0, 4.0]))


class TestNonzero:
    def test_basic(self):
        da = DualArray(np.array([0.0, 1.0, 0.0, 3.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        result = np.nonzero(da)
        expected = np.nonzero(da.primal)
        np.testing.assert_array_equal(result[0], expected[0])

    def test_2d(self):
        p = np.array([[0.0, 1.0], [2.0, 0.0]])
        t = np.ones_like(p) * 0.1
        da = DualArray(p, t)
        result = np.nonzero(da)
        expected = np.nonzero(p)
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])


class TestFlatnonzero:
    def test_basic(self):
        da = DualArray(np.array([0.0, 1.0, 0.0, 3.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        result = np.flatnonzero(da)
        np.testing.assert_array_equal(result, np.flatnonzero(da.primal))


class TestExtract:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        cond = np.array([True, False, True, False])
        result = np.extract(cond, da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([1.0, 3.0]))
        np.testing.assert_allclose(result.tangent, np.array([0.1, 0.3]))
