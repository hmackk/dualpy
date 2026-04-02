import numpy as np
import pytest

from dualpy.core import DualArray

pytestmark = pytest.mark.unit


@pytest.fixture(
    params=[
        pytest.param(np.array(2.7), id="scalar"),
        pytest.param(np.array([-1.3, 0.0, 2.7, -0.5]), id="vector"),
    ]
)
def x_val(request):
    return request.param


class TestSign:
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.sign(da)
        np.testing.assert_array_equal(result.primal, np.sign(x_val))

    def test_tangent_zero(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val) * 3.0)
        result = np.sign(da)
        np.testing.assert_array_equal(result.tangent, np.zeros_like(x_val))

    def test_returns_dual(self):
        da = DualArray(np.array(5.0), np.array(1.0))
        result = np.sign(da)
        assert isinstance(result, DualArray)

    def test_at_zero(self):
        da = DualArray(np.array(0.0), np.array(1.0))
        result = np.sign(da)
        np.testing.assert_allclose(result.primal, 0.0)
        np.testing.assert_allclose(result.tangent, 0.0)


class TestHeaviside:
    def test_primal(self):
        x = np.array([-1.0, 0.0, 1.0])
        da = DualArray(x, np.ones_like(x))
        h = DualArray(np.array([0.5, 0.5, 0.5]), np.zeros(3))
        result = np.heaviside(da, h)
        np.testing.assert_array_equal(result.primal, np.heaviside(x, 0.5))

    def test_tangent_zero(self):
        x = np.array([-1.0, 0.0, 1.0])
        da = DualArray(x, np.array([1.0, 2.0, 3.0]))
        h = DualArray(np.array([0.5, 0.5, 0.5]), np.zeros(3))
        result = np.heaviside(da, h)
        np.testing.assert_array_equal(result.tangent, np.zeros(3))

    def test_returns_dual(self):
        da = DualArray(np.array(1.0), np.array(1.0))
        h = DualArray(np.array(0.5), np.array(0.0))
        result = np.heaviside(da, h)
        assert isinstance(result, DualArray)


class TestFloor:
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.floor(da)
        np.testing.assert_array_equal(result.primal, np.floor(x_val))

    def test_tangent_zero(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val) * 5.0)
        result = np.floor(da)
        np.testing.assert_array_equal(result.tangent, np.zeros_like(x_val))

    def test_returns_dual(self):
        da = DualArray(np.array(3.7), np.array(1.0))
        result = np.floor(da)
        assert isinstance(result, DualArray)

    def test_at_integer(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        result = np.floor(da)
        np.testing.assert_allclose(result.primal, 3.0)
        np.testing.assert_allclose(result.tangent, 0.0)


class TestCeil:
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.ceil(da)
        np.testing.assert_array_equal(result.primal, np.ceil(x_val))

    def test_tangent_zero(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val) * 5.0)
        result = np.ceil(da)
        np.testing.assert_array_equal(result.tangent, np.zeros_like(x_val))

    def test_returns_dual(self):
        da = DualArray(np.array(3.2), np.array(1.0))
        result = np.ceil(da)
        assert isinstance(result, DualArray)


class TestTrunc:
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.trunc(da)
        np.testing.assert_array_equal(result.primal, np.trunc(x_val))

    def test_tangent_zero(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val) * 5.0)
        result = np.trunc(da)
        np.testing.assert_array_equal(result.tangent, np.zeros_like(x_val))

    def test_returns_dual(self):
        da = DualArray(np.array(-3.7), np.array(1.0))
        result = np.trunc(da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, -3.0)


class TestRint:
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.rint(da)
        np.testing.assert_array_equal(result.primal, np.rint(x_val))

    def test_tangent_zero(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val) * 5.0)
        result = np.rint(da)
        np.testing.assert_array_equal(result.tangent, np.zeros_like(x_val))

    def test_returns_dual(self):
        da = DualArray(np.array(2.7), np.array(1.0))
        result = np.rint(da)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, 3.0)
