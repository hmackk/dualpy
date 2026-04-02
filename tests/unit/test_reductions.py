import numpy as np
import pytest
from dualpy.core import DualArray

from tests.conftest import (
    assert_dual_close,
    assert_tangent_close_to_fd,
    assert_tangent_finite,
    finite_diff_jvp,
)

pytestmark = pytest.mark.unit


class TestSum:
    def test_sum_scalar_result(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.sum(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array(6.0), np.array(0.6))

    def test_sum_with_axis(self):
        da = DualArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        result = np.sum(da, axis=0)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([4.0, 6.0]), np.array([0.4, 0.6]))


class TestMean:
    def test_mean_scalar_result(self):
        da = DualArray(np.array([2.0, 4.0, 6.0]), np.array([0.2, 0.4, 0.6]))
        result = np.mean(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array(4.0), np.array(0.4))

    def test_mean_with_axis(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.mean(da, axis=1)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.mean(x, axis=1))
        np.testing.assert_allclose(result.tangent, np.mean(v, axis=1))

    def test_mean_vs_finite_diff(self):
        x = np.array([1.0, 3.0, 5.0])
        v = np.array([0.1, -0.2, 0.3])
        assert_tangent_close_to_fd(np.mean, x, v)


class TestProd:
    def test_prod_scalar_result(self):
        da = DualArray(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
        result = np.prod(da)
        assert isinstance(result, DualArray)
        expected_primal = np.array(6.0)
        expected_tangent = np.array(1.0 * 3.0 + 2.0 * 1.0)
        assert_dual_close(result, expected_primal, expected_tangent)

    def test_prod_with_zero_element(self):
        da = DualArray(np.array([0.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0]))
        result = np.prod(da)
        assert result.primal == 0.0
        assert np.isfinite(result.tangent)
        np.testing.assert_allclose(result.tangent, 3.0 * 4.0 * 1.0)

    def test_prod_vs_finite_diff(self):
        x = np.array([2.0, 0.5, 5.0])
        v = np.array([1.0, 1.0, 1.0])
        assert_tangent_close_to_fd(np.prod, x, v)

    def test_prod_with_zero_vs_finite_diff(self):
        x = np.array([2.0, 0.0, 5.0])
        v = np.array([1.0, 1.0, 1.0])
        fd = finite_diff_jvp(np.prod, x, v)
        da = DualArray(x, v)
        result = np.prod(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_prod_with_axis(self):
        x = np.array([[2.0, 3.0], [4.0, 5.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.prod(da, axis=0)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.prod(x, axis=0))
        fd = finite_diff_jvp(lambda z: np.prod(z, axis=0), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_prod_with_axis_keepdims(self):
        x = np.array([[2.0, 3.0], [4.0, 5.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.prod(da, axis=1, keepdims=True)
        assert isinstance(result, DualArray)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.primal, np.prod(x, axis=1, keepdims=True))
        fd = finite_diff_jvp(lambda z: np.prod(z, axis=1, keepdims=True), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestMax:
    def test_max_scalar_result(self):
        da = DualArray(np.array([1.0, 5.0, 3.0]), np.array([0.1, 0.5, 0.3]))
        result = np.max(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array(5.0), np.array(0.5))

    def test_max_with_axis(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.max(da, axis=0)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([3.0, 5.0]), np.array([0.3, 0.5]))

    def test_max_with_axis1(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.max(da, axis=1)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([5.0, 3.0]), np.array([0.5, 0.3]))

    def test_max_with_axis_keepdims(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.max(da, axis=1, keepdims=True)
        assert isinstance(result, DualArray)
        assert result.shape == (2, 1)
        assert_dual_close(
            result,
            np.array([[5.0], [3.0]]),
            np.array([[0.5], [0.3]]),
        )

    def test_max_vs_finite_diff(self):
        x = np.array([1.0, 5.0, 3.0])
        v = np.array([0.1, -0.2, 0.3])
        assert_tangent_close_to_fd(np.max, x, v)


class TestMin:
    def test_min_scalar_result(self):
        da = DualArray(np.array([1.0, 5.0, 3.0]), np.array([0.1, 0.5, 0.3]))
        result = np.min(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array(1.0), np.array(0.1))

    def test_min_with_axis(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.min(da, axis=0)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([1.0, 2.0]), np.array([0.1, 0.2]))

    def test_min_with_axis1(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.min(da, axis=1)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([1.0, 2.0]), np.array([0.1, 0.2]))

    def test_min_with_axis_keepdims(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.min(da, axis=0, keepdims=True)
        assert isinstance(result, DualArray)
        assert result.shape == (1, 2)
        assert_dual_close(
            result,
            np.array([[1.0, 2.0]]),
            np.array([[0.1, 0.2]]),
        )

    def test_min_vs_finite_diff(self):
        x = np.array([1.0, 5.0, 3.0])
        v = np.array([0.1, -0.2, 0.3])
        assert_tangent_close_to_fd(np.min, x, v)


class TestVar:
    def test_scalar_result(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
        result = np.var(da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.var([1.0, 2.0, 3.0]), rtol=1e-7)

    def test_constant_tangent(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]))
        result = np.var(da)
        np.testing.assert_allclose(result.tangent, 0.0, atol=1e-12)

    def test_var_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, -0.2, 0.3])
        assert_tangent_close_to_fd(np.var, x, v)

    def test_var_ddof1(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.array([0.1, -0.2, 0.3, -0.1])
        fd = finite_diff_jvp(lambda z: np.var(z, ddof=1), x, v)
        da = DualArray(x, v)
        result = np.var(da, ddof=1)
        np.testing.assert_allclose(result.primal, np.var(x, ddof=1), rtol=1e-7)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_var_ddof1_vs_finite_diff(self):
        x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        assert_tangent_close_to_fd(np.var, x, ddof=1)

    def test_var_keepdims(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.var(da, axis=0, keepdims=True)
        assert isinstance(result, DualArray)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result.primal, np.var(x, axis=0, keepdims=True))
        fd = finite_diff_jvp(lambda z: np.var(z, axis=0, keepdims=True), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestStd:
    def test_scalar_result(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0]))
        result = np.std(da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.std([1.0, 2.0, 3.0]), rtol=1e-7)

    def test_std_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 0.0, 0.0])
        assert_tangent_close_to_fd(np.std, x, v)

    def test_std_ddof1(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.array([0.1, -0.2, 0.3, -0.1])
        fd = finite_diff_jvp(lambda z: np.std(z, ddof=1), x, v)
        da = DualArray(x, v)
        result = np.std(da, ddof=1)
        np.testing.assert_allclose(result.primal, np.std(x, ddof=1), rtol=1e-7)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_std_constant_input(self):
        da = DualArray(np.array([3.0, 3.0, 3.0]), np.array([1.0, 1.0, 1.0]))
        result = np.std(da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, 0.0, atol=1e-12)
        assert_tangent_finite(result)

    def test_std_keepdims(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.std(da, axis=1, keepdims=True)
        assert isinstance(result, DualArray)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.primal, np.std(x, axis=1, keepdims=True))
        fd = finite_diff_jvp(lambda z: np.std(z, axis=1, keepdims=True), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestCumsum:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.cumsum(da)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 3.0, 6.0]),
            np.array([0.1, 0.3, 0.6]),
        )

    def test_cumsum_vs_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, 0.2, 0.3])
        assert_tangent_close_to_fd(np.cumsum, x, v)

    def test_cumsum_with_axis(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.cumsum(da, axis=0)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.cumsum(x, axis=0))
        np.testing.assert_allclose(result.tangent, np.cumsum(v, axis=0))

    def test_cumsum_with_axis1(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.cumsum(da, axis=1)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.cumsum(x, axis=1))
        np.testing.assert_allclose(result.tangent, np.cumsum(v, axis=1))


class TestCumprod:
    def test_basic(self):
        da = DualArray(np.array([2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0]))
        result = np.cumprod(da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([2.0, 6.0, 24.0]), rtol=1e-7)
        np.testing.assert_allclose(result.tangent[0], 1.0, rtol=1e-7)
        np.testing.assert_allclose(result.tangent[1], 1.0 * 3.0 + 2.0 * 1.0, rtol=1e-7)
        np.testing.assert_allclose(
            result.tangent[2],
            result.tangent[1] * 4.0 + 6.0 * 1.0,
            rtol=1e-7,
        )

    def test_cumprod_vs_finite_diff(self):
        x = np.array([2.0, 3.0, 4.0])
        v = np.array([1.0, 1.0, 1.0])
        assert_tangent_close_to_fd(np.cumprod, x, v)

    def test_cumprod_with_axis(self):
        x = np.array([[2.0, 3.0], [4.0, 5.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.cumprod(da, axis=0)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.cumprod(x, axis=0))
        fd = finite_diff_jvp(lambda z: np.cumprod(z, axis=0), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_cumprod_with_axis1(self):
        x = np.array([[2.0, 3.0], [4.0, 5.0]])
        v = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, v)
        result = np.cumprod(da, axis=1)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.cumprod(x, axis=1))
        fd = finite_diff_jvp(lambda z: np.cumprod(z, axis=1), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestArgmax:
    def test_basic(self):
        da = DualArray(np.array([1.0, 5.0, 3.0]), np.array([0.1, 0.5, 0.3]))
        result = np.argmax(da)
        assert result == 1
        assert not isinstance(result, DualArray)

    def test_with_axis(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.argmax(da, axis=0)
        np.testing.assert_array_equal(result, np.argmax(da.primal, axis=0))

    def test_with_axis1(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.argmax(da, axis=1)
        np.testing.assert_array_equal(result, np.argmax(da.primal, axis=1))


class TestArgmin:
    def test_basic(self):
        da = DualArray(np.array([1.0, 5.0, 3.0]), np.array([0.1, 0.5, 0.3]))
        result = np.argmin(da)
        assert result == 0
        assert not isinstance(result, DualArray)

    def test_with_axis(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.argmin(da, axis=0)
        np.testing.assert_array_equal(result, np.argmin(da.primal, axis=0))

    def test_with_axis1(self):
        da = DualArray(
            np.array([[1.0, 5.0], [3.0, 2.0]]),
            np.array([[0.1, 0.5], [0.3, 0.2]]),
        )
        result = np.argmin(da, axis=1)
        np.testing.assert_array_equal(result, np.argmin(da.primal, axis=1))


class TestNansum:
    def test_basic(self):
        x = np.array([1.0, np.nan, 3.0])
        t = np.array([0.1, 0.2, 0.3])
        da = DualArray(x, t)
        result = np.nansum(da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.nansum(x))
        np.testing.assert_allclose(result.tangent, 0.1 + 0.3)

    def test_no_nans(self):
        x = np.array([1.0, 2.0, 3.0])
        t = np.array([0.1, 0.2, 0.3])
        da = DualArray(x, t)
        result = np.nansum(da)
        np.testing.assert_allclose(result.primal, 6.0)
        np.testing.assert_allclose(result.tangent, 0.6)

    def test_with_axis(self):
        x = np.array([[1.0, np.nan], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, t)
        result = np.nansum(da, axis=0)
        np.testing.assert_allclose(result.primal, np.nansum(x, axis=0))


class TestNanmean:
    def test_basic(self):
        x = np.array([1.0, np.nan, 3.0])
        t = np.array([0.1, 0.2, 0.3])
        da = DualArray(x, t)
        result = np.nanmean(da)
        np.testing.assert_allclose(result.primal, np.nanmean(x))
        np.testing.assert_allclose(result.tangent, (0.1 + 0.3) / 2.0)

    def test_finite_diff(self):
        x = np.array([1.0, np.nan, 3.0, 4.0])
        v = np.array([1.0, 0.0, 0.5, 0.2])
        da = DualArray(x, v)
        result = np.nanmean(da)
        fd = finite_diff_jvp(lambda z: np.nanmean(z), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestNanvar:
    def test_basic(self):
        x = np.array([1.0, np.nan, 3.0, 5.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, t)
        result = np.nanvar(da)
        np.testing.assert_allclose(result.primal, np.nanvar(x))

    def test_finite_diff(self):
        x = np.array([1.0, np.nan, 3.0, 5.0])
        v = np.array([1.0, 0.0, 0.5, 0.2])
        da = DualArray(x, v)
        result = np.nanvar(da)
        fd = finite_diff_jvp(lambda z: np.nanvar(z), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestNanstd:
    def test_basic(self):
        x = np.array([1.0, np.nan, 3.0, 5.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, t)
        result = np.nanstd(da)
        np.testing.assert_allclose(result.primal, np.nanstd(x))

    def test_finite_diff(self):
        x = np.array([1.0, np.nan, 3.0, 5.0])
        v = np.array([1.0, 0.0, 0.5, 0.2])
        da = DualArray(x, v)
        result = np.nanstd(da)
        fd = finite_diff_jvp(lambda z: np.nanstd(z), x, v)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestNanmax:
    def test_basic(self):
        x = np.array([1.0, np.nan, 5.0, 3.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, t)
        result = np.nanmax(da)
        np.testing.assert_allclose(result.primal, 5.0)
        np.testing.assert_allclose(result.tangent, 0.3)

    def test_with_axis(self):
        x = np.array([[1.0, np.nan], [3.0, 2.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, t)
        result = np.nanmax(da, axis=0)
        np.testing.assert_allclose(result.primal, np.nanmax(x, axis=0))


class TestNanmin:
    def test_basic(self):
        x = np.array([5.0, np.nan, 1.0, 3.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, t)
        result = np.nanmin(da)
        np.testing.assert_allclose(result.primal, 1.0)
        np.testing.assert_allclose(result.tangent, 0.3)

    def test_with_axis(self):
        x = np.array([[5.0, np.nan], [3.0, 2.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(x, t)
        result = np.nanmin(da, axis=0)
        np.testing.assert_allclose(result.primal, np.nanmin(x, axis=0))


class TestAverage:
    def test_unweighted(self):
        x = np.array([1.0, 2.0, 3.0])
        t = np.array([0.1, 0.2, 0.3])
        da = DualArray(x, t)
        result = np.average(da)
        np.testing.assert_allclose(result.primal, 2.0)
        np.testing.assert_allclose(result.tangent, 0.2)

    def test_weighted(self):
        x = np.array([1.0, 2.0, 3.0])
        t = np.array([0.1, 0.2, 0.3])
        w = np.array([1.0, 2.0, 1.0])
        da = DualArray(x, t)
        result = np.average(da, weights=w)
        np.testing.assert_allclose(result.primal, np.average(x, weights=w))

    def test_weighted_finite_diff(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 2.0, 1.0])
        v = np.array([1.0, 0.5, 0.2])
        fd = finite_diff_jvp(lambda z: np.average(z, weights=w), x, v)
        da = DualArray(x, v)
        result = np.average(da, weights=w)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_with_returned(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 2.0, 1.0])
        da = DualArray(x, np.ones_like(x))
        result, sw = np.average(da, weights=w, returned=True)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(sw, np.sum(w))


class TestMedian:
    def test_odd_size(self):
        x = np.array([3.0, 1.0, 2.0])
        t = np.array([0.3, 0.1, 0.2])
        da = DualArray(x, t)
        result = np.median(da)
        np.testing.assert_allclose(result.primal, 2.0)
        np.testing.assert_allclose(result.tangent, 0.2)

    def test_even_size(self):
        x = np.array([4.0, 1.0, 3.0, 2.0])
        t = np.array([0.4, 0.1, 0.3, 0.2])
        da = DualArray(x, t)
        result = np.median(da)
        np.testing.assert_allclose(result.primal, 2.5)
        np.testing.assert_allclose(result.tangent, (0.2 + 0.3) / 2.0)

    def test_with_axis(self):
        x = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        t = np.ones_like(x) * 0.1
        da = DualArray(x, t)
        result = np.median(da, axis=1)
        np.testing.assert_allclose(result.primal, np.median(x, axis=1))


class TestAll:
    def test_true(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.all(da)
        assert result == True  # noqa: E712

    def test_false(self):
        da = DualArray(np.array([1.0, 0.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.all(da)
        assert result == False  # noqa: E712

    def test_with_axis(self):
        da = DualArray(
            np.array([[1.0, 0.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        result = np.all(da, axis=0)
        np.testing.assert_array_equal(result, np.all(da.primal, axis=0))

    def test_returns_plain(self):
        da = DualArray(np.array([1.0]), np.array([0.1]))
        result = np.all(da)
        assert not isinstance(result, DualArray)


class TestAny:
    def test_true(self):
        da = DualArray(np.array([0.0, 0.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.any(da)
        assert result == True  # noqa: E712

    def test_false(self):
        da = DualArray(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.2, 0.3]))
        result = np.any(da)
        assert result == False  # noqa: E712

    def test_returns_plain(self):
        da = DualArray(np.array([1.0]), np.array([0.1]))
        result = np.any(da)
        assert not isinstance(result, DualArray)


class TestCountNonzero:
    def test_basic(self):
        da = DualArray(np.array([0.0, 1.0, 0.0, 3.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        result = np.count_nonzero(da)
        assert result == 2

    def test_with_axis(self):
        da = DualArray(
            np.array([[0.0, 1.0], [2.0, 0.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        result = np.count_nonzero(da, axis=0)
        np.testing.assert_array_equal(result, np.array([1, 1]))
