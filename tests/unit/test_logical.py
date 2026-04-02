import numpy as np
import pytest
from dualpy.core import DualArray

pytestmark = pytest.mark.unit


class TestLogicalAnd:
    def test_result_matches_numpy(self):
        a = DualArray(np.array([1.0, 0.0, 1.0]), np.array([0.1, 0.2, 0.3]))
        b = DualArray(np.array([1.0, 1.0, 0.0]), np.array([0.4, 0.5, 0.6]))
        result = np.logical_and(a, b)
        np.testing.assert_array_equal(result, np.logical_and(a.primal, b.primal))

    def test_returns_plain_array(self):
        a = DualArray(np.array([1.0, 0.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([1.0, 1.0]), np.array([0.3, 0.4]))
        result = np.logical_and(a, b)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, DualArray)


class TestLogicalOr:
    def test_result_matches_numpy(self):
        a = DualArray(np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.2, 0.3]))
        b = DualArray(np.array([0.0, 1.0, 0.0]), np.array([0.4, 0.5, 0.6]))
        result = np.logical_or(a, b)
        np.testing.assert_array_equal(result, np.logical_or(a.primal, b.primal))

    def test_returns_plain_array(self):
        a = DualArray(np.array([1.0, 0.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([0.0, 1.0]), np.array([0.3, 0.4]))
        result = np.logical_or(a, b)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, DualArray)


class TestLogicalXor:
    def test_result_matches_numpy(self):
        a = DualArray(np.array([1.0, 0.0, 1.0]), np.array([0.1, 0.2, 0.3]))
        b = DualArray(np.array([1.0, 1.0, 0.0]), np.array([0.4, 0.5, 0.6]))
        result = np.logical_xor(a, b)
        np.testing.assert_array_equal(result, np.logical_xor(a.primal, b.primal))

    def test_returns_plain_array(self):
        a = DualArray(np.array([1.0]), np.array([0.1]))
        b = DualArray(np.array([0.0]), np.array([0.2]))
        result = np.logical_xor(a, b)
        assert not isinstance(result, DualArray)


class TestLogicalNot:
    def test_result_matches_numpy(self):
        a = DualArray(np.array([1.0, 0.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.logical_not(a)
        np.testing.assert_array_equal(result, np.logical_not(a.primal))

    def test_returns_plain_array(self):
        a = DualArray(np.array([1.0, 0.0]), np.array([0.1, 0.2]))
        result = np.logical_not(a)
        assert not isinstance(result, DualArray)


class TestIsnan:
    def test_no_nans(self):
        a = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.isnan(a)
        np.testing.assert_array_equal(result, np.array([False, False, False]))

    def test_with_nans(self):
        a = DualArray(np.array([1.0, np.nan, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.isnan(a)
        np.testing.assert_array_equal(result, np.array([False, True, False]))

    def test_returns_plain_array(self):
        a = DualArray(np.array([1.0, np.nan]), np.array([0.1, 0.2]))
        result = np.isnan(a)
        assert not isinstance(result, DualArray)


class TestIsinf:
    def test_no_inf(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.isinf(a)
        np.testing.assert_array_equal(result, np.array([False, False]))

    def test_with_inf(self):
        a = DualArray(np.array([1.0, np.inf, -np.inf]), np.array([0.1, 0.2, 0.3]))
        result = np.isinf(a)
        np.testing.assert_array_equal(result, np.array([False, True, True]))

    def test_returns_plain_array(self):
        a = DualArray(np.array([np.inf]), np.array([0.1]))
        result = np.isinf(a)
        assert not isinstance(result, DualArray)


class TestIsfinite:
    def test_all_finite(self):
        a = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.isfinite(a)
        np.testing.assert_array_equal(result, np.array([True, True, True]))

    def test_with_non_finite(self):
        a = DualArray(np.array([1.0, np.inf, np.nan]), np.array([0.1, 0.2, 0.3]))
        result = np.isfinite(a)
        np.testing.assert_array_equal(result, np.array([True, False, False]))

    def test_returns_plain_array(self):
        a = DualArray(np.array([1.0, np.nan]), np.array([0.1, 0.2]))
        result = np.isfinite(a)
        assert not isinstance(result, DualArray)


class TestSignbit:
    def test_positive(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.signbit(a)
        np.testing.assert_array_equal(result, np.array([False, False]))

    def test_negative(self):
        a = DualArray(np.array([-1.0, -2.0]), np.array([0.1, 0.2]))
        result = np.signbit(a)
        np.testing.assert_array_equal(result, np.array([True, True]))

    def test_mixed(self):
        a = DualArray(np.array([-3.0, 0.0, 5.0]), np.array([0.1, 0.2, 0.3]))
        result = np.signbit(a)
        np.testing.assert_array_equal(result, np.signbit(a.primal))

    def test_returns_plain_array(self):
        a = DualArray(np.array([-1.0]), np.array([0.1]))
        result = np.signbit(a)
        assert not isinstance(result, DualArray)

    def test_negative_zero(self):
        a = DualArray(np.array([-0.0]), np.array([1.0]))
        result = np.signbit(a)
        np.testing.assert_array_equal(result, np.array([True]))
