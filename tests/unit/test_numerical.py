import numpy as np
import pytest
from dualpy.core import DualArray

from tests.conftest import assert_dual_close, finite_diff_jvp

pytestmark = pytest.mark.unit


class TestDiff:
    def test_basic(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, t)
        result = np.diff(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.diff(x), np.diff(t))

    def test_n2(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, t)
        result = np.diff(da, n=2)
        assert_dual_close(result, np.diff(x, n=2), np.diff(t, n=2))

    def test_with_axis(self):
        p = np.array([[1.0, 3.0, 6.0], [2.0, 5.0, 9.0]])
        t = np.ones_like(p) * 0.1
        da = DualArray(p, t)
        result = np.diff(da, axis=0)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.diff(p, axis=0), np.diff(t, axis=0))

    def test_finite_diff(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        v = np.array([0.1, -0.2, 0.3, -0.1])
        fd = finite_diff_jvp(lambda z: np.diff(z), x, v)
        da = DualArray(x, v)
        result = np.diff(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestConvolve:
    def test_full(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.0])
        ta = np.array([0.1, 0.2, 0.3])
        da = DualArray(a, ta)
        result = np.convolve(da, b)
        np.testing.assert_allclose(result.primal, np.convolve(a, b))

    def test_vs_finite_diff_a(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.0])
        v = np.array([0.1, -0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.convolve(z, b), a, v)
        da = DualArray(a, v)
        result = np.convolve(da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_same_mode(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([0.5, 1.0, 0.5])
        da = DualArray(a, np.ones_like(a))
        result = np.convolve(da, b, mode="same")
        np.testing.assert_allclose(result.primal, np.convolve(a, b, mode="same"))


class TestCorrelate:
    def test_valid(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 0.5])
        da = DualArray(a, np.ones_like(a))
        result = np.correlate(da, b)
        np.testing.assert_allclose(result.primal, np.correlate(a, b))

    def test_vs_finite_diff(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 0.5])
        v = np.array([0.1, -0.2, 0.3, 0.0])
        fd = finite_diff_jvp(lambda z: np.correlate(z, b), a, v)
        da = DualArray(a, v)
        result = np.correlate(da, b)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestInterp:
    def test_basic(self):
        xp = np.array([0.0, 1.0, 2.0])
        fp = np.array([0.0, 1.0, 4.0])
        x = np.array([0.5, 1.5])
        dfp = DualArray(fp, np.array([0.1, 0.2, 0.3]))
        result = np.interp(x, xp, dfp)
        np.testing.assert_allclose(result.primal, np.interp(x, xp, fp))

    def test_tangent_fp(self):
        xp = np.array([0.0, 1.0, 2.0])
        fp = np.array([0.0, 1.0, 4.0])
        x = np.array([0.5, 1.5])
        v = np.array([0.1, 0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.interp(x, xp, z), fp, v)
        dfp = DualArray(fp, v)
        result = np.interp(x, xp, dfp)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-4)


class TestTrapezoid:
    def test_basic(self):
        y = np.array([0.0, 1.0, 4.0])
        t = np.array([0.1, 0.2, 0.3])
        da = DualArray(y, t)
        result = np.trapezoid(da)
        np.testing.assert_allclose(result.primal, np.trapezoid(y))
        np.testing.assert_allclose(result.tangent, np.trapezoid(t))

    def test_with_dx(self):
        y = np.array([0.0, 1.0, 4.0, 9.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(y, t)
        result = np.trapezoid(da, dx=0.5)
        np.testing.assert_allclose(result.primal, np.trapezoid(y, dx=0.5))
        np.testing.assert_allclose(result.tangent, np.trapezoid(t, dx=0.5))

    def test_finite_diff(self):
        y = np.array([0.0, 1.0, 4.0, 9.0])
        v = np.array([0.1, -0.2, 0.3, 0.0])
        fd = finite_diff_jvp(lambda z: np.trapezoid(z), y, v)
        da = DualArray(y, v)
        result = np.trapezoid(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_with_x_tangent(self):
        y = np.array([0.0, 1.0, 4.0, 9.0])
        x = np.array([0.0, 0.5, 1.5, 3.0])
        vy = np.array([0.1, -0.2, 0.3, 0.0])
        vx = np.array([0.0, 0.1, -0.1, 0.2])
        fd = finite_diff_jvp(
            lambda z: np.trapezoid(z[:4], x=z[4:]),
            np.concatenate([y, x]),
            np.concatenate([vy, vx]),
        )
        dy = DualArray(y, vy)
        dx = DualArray(x, vx)
        result = np.trapezoid(dy, dx)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_with_x_only_y_tangent(self):
        y = np.array([1.0, 2.0, 3.0])
        x = np.array([0.0, 1.0, 3.0])
        v = np.array([0.1, -0.2, 0.3])
        fd = finite_diff_jvp(lambda z: np.trapezoid(z, x=x), y, v)
        da = DualArray(y, v)
        result = np.trapezoid(da, x)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestSinc:
    def test_basic(self):
        x = np.array([0.5, 1.0, 1.5])
        t = np.ones_like(x)
        da = DualArray(x, t)
        result = np.sinc(da)
        np.testing.assert_allclose(result.primal, np.sinc(x), rtol=1e-7)

    def test_at_zero(self):
        da = DualArray(np.array(0.0), np.array(1.0))
        result = np.sinc(da)
        np.testing.assert_allclose(result.primal, 1.0)
        assert np.isfinite(result.tangent)
        np.testing.assert_allclose(result.tangent, 0.0, atol=1e-10)

    def test_at_integer(self):
        da = DualArray(np.array(1.0), np.array(1.0))
        result = np.sinc(da)
        np.testing.assert_allclose(result.primal, 0.0, atol=1e-15)

    def test_finite_diff(self):
        x = np.array([0.3, 0.7, 1.3])
        v = np.array([1.0, 0.5, 0.2])
        fd = finite_diff_jvp(lambda z: np.sinc(z), x, v)
        da = DualArray(x, v)
        result = np.sinc(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-4)


class TestNumpyGradient:
    def test_1d(self):
        x = np.array([1.0, 4.0, 9.0, 16.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, t)
        result = np.gradient(da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.gradient(x))
        np.testing.assert_allclose(result.tangent, np.gradient(t))

    def test_2d(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = np.ones_like(x) * 0.1
        da = DualArray(x, t)
        result = np.gradient(da)
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, DualArray)

    def test_with_spacing(self):
        x = np.array([0.0, 1.0, 4.0, 9.0])
        t = np.array([0.1, 0.2, 0.3, 0.4])
        da = DualArray(x, t)
        result = np.gradient(da, 0.5)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.gradient(x, 0.5))
