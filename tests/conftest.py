import numpy as np
import pytest

import dualpy  # noqa: F401 — triggers ufunc and routine registration
from dualpy.core import DualArray


def assert_dual_close(
    result: DualArray,
    expected_primal,
    expected_tangent,
    rtol: float = 1e-7,
):
    np.testing.assert_allclose(
        result.primal,
        expected_primal,
        rtol=rtol,
        err_msg="primal mismatch",
    )
    np.testing.assert_allclose(
        result.tangent,
        expected_tangent,
        rtol=rtol,
        err_msg="tangent mismatch",
    )


def finite_diff_jvp(f, x, v, eps=1e-7):
    """Approximate the JVP (J @ v) via central finite differences."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    return (f(x + eps * v) - f(x - eps * v)) / (2 * eps)


def assert_tangent_close_to_fd(np_func, x, v=None, rtol=1e-5, **kwargs):
    """Assert that the DualArray tangent matches finite differences."""
    if v is None:
        v = np.ones_like(x)
    fd = finite_diff_jvp(lambda z: np_func(z, **kwargs), x, v)
    da = DualArray(x, v)
    result = np_func(da, **kwargs)
    np.testing.assert_allclose(result.tangent, fd, rtol=rtol)


def assert_tangent_finite(result):
    """Assert that all tangent values are finite (no inf/nan)."""
    assert np.all(np.isfinite(result.tangent)), (
        f"tangent contains non-finite values: {result.tangent}"
    )


@pytest.fixture
def scalar_dual():
    return DualArray(np.array(2.0), np.array(1.0))


@pytest.fixture
def vector_dual():
    return DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 2.0]))


@pytest.fixture
def matrix_dual():
    primal = np.array([[1.0, 2.0], [3.0, 4.0]])
    tangent = np.array([[0.1, 0.2], [0.3, 0.4]])
    return DualArray(primal, tangent)


@pytest.fixture
def random_vector():
    rng = np.random.default_rng(42)
    return rng.standard_normal(5)
