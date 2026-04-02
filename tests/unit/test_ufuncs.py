import numpy as np
import pytest
from dualpy.core import DualArray

from tests.conftest import assert_dual_close, assert_tangent_finite, finite_diff_jvp

pytestmark = pytest.mark.unit


@pytest.fixture(
    params=[
        pytest.param(np.array(2.0), id="scalar"),
        pytest.param(np.array([1.0, 2.0, 3.0]), id="vector"),
    ]
)
def x_val(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(np.array(3.0), id="scalar"),
        pytest.param(np.array([4.0, 5.0, 6.0]), id="vector"),
    ]
)
def y_val(request):
    return request.param


class TestAddition:
    def test_dual_dual(self, x_val, y_val):
        if x_val.shape != y_val.shape:
            pytest.skip("shape mismatch for this parametrize combo")
        xt = np.ones_like(x_val)
        yt = np.ones_like(y_val) * 0.5
        dx = DualArray(x_val, xt)
        dy = DualArray(y_val, yt)
        result = dx + dy
        assert_dual_close(result, x_val + y_val, xt + yt)

    def test_finite_diff(self):
        x = np.array([1.0, 2.0])
        v = np.array([1.0, 0.0])

        def f(x):
            return x + np.array([10.0, 20.0])

        fd = finite_diff_jvp(f, x, v)
        da = DualArray(x, v) + DualArray(np.array([10.0, 20.0]))
        np.testing.assert_allclose(da.tangent, fd, rtol=1e-5)

    def test_dual_plus_scalar(self):
        da = DualArray(np.array(2.0), np.array(1.0))
        result = da + 5.0
        assert_dual_close(result, np.array(7.0), np.array(1.0))

    def test_scalar_plus_dual(self):
        da = DualArray(np.array(2.0), np.array(1.0))
        result = 5.0 + da
        assert_dual_close(result, np.array(7.0), np.array(1.0))

    def test_dual_plus_ndarray(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        result = da + np.array([10.0, 20.0])
        assert_dual_close(result, np.array([11.0, 22.0]), np.array([1.0, 1.0]))


class TestSubtraction:
    def test_dual_dual(self, x_val, y_val):
        if x_val.shape != y_val.shape:
            pytest.skip("shape mismatch for this parametrize combo")
        xt = np.ones_like(x_val)
        yt = np.ones_like(y_val) * 0.5
        dx = DualArray(x_val, xt)
        dy = DualArray(y_val, yt)
        result = dx - dy
        assert_dual_close(result, x_val - y_val, xt - yt)

    def test_dual_minus_scalar(self):
        da = DualArray(np.array(5.0), np.array(1.0))
        result = da - 2.0
        assert_dual_close(result, np.array(3.0), np.array(1.0))

    def test_scalar_minus_dual(self):
        da = DualArray(np.array(2.0), np.array(1.0))
        result = 5.0 - da
        assert_dual_close(result, np.array(3.0), np.array(-1.0))


class TestMultiplication:
    def test_dual_dual(self, x_val, y_val):
        if x_val.shape != y_val.shape:
            pytest.skip("shape mismatch for this parametrize combo")
        xt = np.ones_like(x_val) * 0.5
        yt = np.ones_like(y_val) * 2.0
        dx = DualArray(x_val, xt)
        dy = DualArray(y_val, yt)
        result = dx * dy
        expected_tangent = xt * y_val + x_val * yt
        assert_dual_close(result, x_val * y_val, expected_tangent)

    def test_finite_diff(self):
        x = np.array(3.0)
        v = np.array(1.0)
        c = np.array(4.0)

        def f(x):
            return x * c

        fd = finite_diff_jvp(f, x, v)
        da = DualArray(x, v) * DualArray(c)
        np.testing.assert_allclose(da.tangent, fd, rtol=1e-5)

    def test_rmul_scalar(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        result = 2.0 * da
        assert_dual_close(result, np.array(6.0), np.array(2.0))

    def test_dual_times_scalar(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        result = da * 2.0
        assert_dual_close(result, np.array(6.0), np.array(2.0))


class TestDivision:
    def test_dual_dual(self):
        x_p, x_t = np.array(6.0), np.array(1.0)
        y_p, y_t = np.array(3.0), np.array(0.5)
        result = DualArray(x_p, x_t) / DualArray(y_p, y_t)
        expected_primal = x_p / y_p
        expected_tangent = (x_t * y_p - x_p * y_t) / (y_p**2)
        assert_dual_close(result, expected_primal, expected_tangent)

    def test_finite_diff(self):
        x = np.array(6.0)
        v = np.array(1.0)
        c = np.array(3.0)

        def f(x):
            return x / c

        fd = finite_diff_jvp(f, x, v)
        da = DualArray(x, v) / DualArray(c)
        np.testing.assert_allclose(da.tangent, fd, rtol=1e-5)

    def test_dual_div_scalar(self):
        da = DualArray(np.array(6.0), np.array(1.0))
        result = da / 3.0
        assert_dual_close(result, np.array(2.0), np.array(1.0 / 3.0))

    def test_scalar_div_dual(self):
        da = DualArray(np.array(2.0), np.array(1.0))
        result = 6.0 / da
        assert_dual_close(result, np.array(3.0), np.array(-6.0 / 4.0))

    def test_vector_division(self):
        x = DualArray(np.array([4.0, 9.0]), np.array([1.0, 1.0]))
        y = DualArray(np.array([2.0, 3.0]), np.array([0.0, 0.0]))
        result = x / y
        assert_dual_close(
            result,
            np.array([2.0, 3.0]),
            np.array([1.0 / 2.0, 1.0 / 3.0]),
        )


class TestPower:
    def test_square(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        result = da**2
        assert_dual_close(result, np.array(9.0), np.array(6.0))

    def test_cube(self):
        da = DualArray(np.array(2.0), np.array(1.0))
        result = da**3
        assert_dual_close(result, np.array(8.0), np.array(12.0))

    def test_finite_diff_square(self):
        x = np.array(3.0)
        v = np.array(1.0)

        def f(x):
            return x**2

        fd = finite_diff_jvp(f, x, v)
        da = DualArray(x, v) ** 2
        np.testing.assert_allclose(da.tangent, fd, rtol=1e-5)

    def test_vector_power(self):
        x = DualArray(np.array([2.0, 3.0]), np.array([1.0, 1.0]))
        result = x**2
        assert_dual_close(
            result,
            np.array([4.0, 9.0]),
            np.array([4.0, 6.0]),
        )

    def test_scalar_pow_dual(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        result = 2.0**da
        expected_primal = np.array(8.0)
        expected_tangent = np.array(8.0 * np.log(2.0))
        assert_dual_close(result, expected_primal, expected_tangent)


class TestNegative:
    def test_scalar(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        result = np.negative(da)
        assert_dual_close(result, np.array(-3.0), np.array(-1.0))

    def test_vector(self):
        da = DualArray(np.array([1.0, -2.0]), np.array([0.5, 0.5]))
        result = np.negative(da)
        assert_dual_close(result, np.array([-1.0, 2.0]), np.array([-0.5, -0.5]))


class TestAbsolute:
    def test_positive(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        result = np.absolute(da)
        assert_dual_close(result, np.array(3.0), np.array(1.0))

    def test_negative(self):
        da = DualArray(np.array(-3.0), np.array(1.0))
        result = np.absolute(da)
        assert_dual_close(result, np.array(3.0), np.array(-1.0))

    def test_vector(self):
        da = DualArray(np.array([-2.0, 3.0]), np.array([1.0, 1.0]))
        result = np.absolute(da)
        assert_dual_close(result, np.array([2.0, 3.0]), np.array([-1.0, 1.0]))


class TestSquare:
    def test_scalar(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        result = np.square(da)
        assert_dual_close(result, np.array(9.0), np.array(6.0))

    def test_finite_diff(self):
        x = np.array(3.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.square, x, v)
        result = np.square(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestCbrt:
    def test_scalar(self):
        x = np.array(8.0)
        t = np.array(1.0)
        da = DualArray(x, t)
        result = np.cbrt(da)
        expected_primal = np.cbrt(x)
        expected_tangent = t / (3 * np.cbrt(x) ** 2)
        assert_dual_close(result, expected_primal, expected_tangent)

    def test_finite_diff(self):
        x = np.array(8.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.cbrt, x, v)
        result = np.cbrt(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestReciprocal:
    def test_scalar(self):
        x = np.array(4.0)
        t = np.array(1.0)
        da = DualArray(x, t)
        result = np.reciprocal(da)
        assert_dual_close(result, np.array(0.25), np.array(-1.0 / 16.0))

    def test_finite_diff(self):
        x = np.array(4.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.reciprocal, x, v)
        result = np.reciprocal(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestMaximum:
    def test_basic(self):
        a = DualArray(np.array([1.0, 5.0]), np.array([0.1, 0.5]))
        b = DualArray(np.array([3.0, 2.0]), np.array([0.3, 0.2]))
        result = np.maximum(a, b)
        assert_dual_close(
            result,
            np.array([3.0, 5.0]),
            np.array([0.3, 0.5]),
        )


class TestMinimum:
    def test_basic(self):
        a = DualArray(np.array([1.0, 5.0]), np.array([0.1, 0.5]))
        b = DualArray(np.array([3.0, 2.0]), np.array([0.3, 0.2]))
        result = np.minimum(a, b)
        assert_dual_close(
            result,
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
        )


SCALAR_VALS = [
    pytest.param(np.array(0.5), id="scalar_0.5"),
    pytest.param(np.array(1.0), id="scalar_1.0"),
]

VECTOR_VALS = [
    pytest.param(np.array([0.3, 0.7, 1.2]), id="vector"),
]


class TestSin:
    @pytest.mark.parametrize("x_val", SCALAR_VALS + VECTOR_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.sin(da)
        np.testing.assert_allclose(result.primal, np.sin(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", SCALAR_VALS + VECTOR_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val) * 2.0
        da = DualArray(x_val, t)
        result = np.sin(da)
        expected_tangent = np.cos(x_val) * t
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    @pytest.mark.parametrize("x_val", SCALAR_VALS)
    def test_tangent_finite_diff(self, x_val):
        v = np.array(1.0)
        fd = finite_diff_jvp(np.sin, x_val, v)
        da = DualArray(x_val, v)
        result = np.sin(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestCos:
    @pytest.mark.parametrize("x_val", SCALAR_VALS + VECTOR_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.cos(da)
        np.testing.assert_allclose(result.primal, np.cos(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", SCALAR_VALS + VECTOR_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val) * 2.0
        da = DualArray(x_val, t)
        result = np.cos(da)
        expected_tangent = -np.sin(x_val) * t
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    @pytest.mark.parametrize("x_val", SCALAR_VALS)
    def test_tangent_finite_diff(self, x_val):
        v = np.array(1.0)
        fd = finite_diff_jvp(np.cos, x_val, v)
        da = DualArray(x_val, v)
        result = np.cos(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestTan:
    @pytest.mark.parametrize("x_val", SCALAR_VALS + VECTOR_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.tan(da)
        np.testing.assert_allclose(result.primal, np.tan(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", SCALAR_VALS + VECTOR_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val)
        da = DualArray(x_val, t)
        result = np.tan(da)
        expected_tangent = (1 + np.tan(x_val) ** 2) * t
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    @pytest.mark.parametrize("x_val", SCALAR_VALS)
    def test_tangent_finite_diff(self, x_val):
        v = np.array(1.0)
        fd = finite_diff_jvp(np.tan, x_val, v)
        da = DualArray(x_val, v)
        result = np.tan(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


ARCSIN_VALS = [
    pytest.param(np.array(0.3), id="scalar_0.3"),
    pytest.param(np.array(-0.5), id="scalar_-0.5"),
    pytest.param(np.array([0.1, 0.4, -0.3]), id="vector"),
]


class TestArcsin:
    @pytest.mark.parametrize("x_val", ARCSIN_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.arcsin(da)
        np.testing.assert_allclose(result.primal, np.arcsin(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", ARCSIN_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val)
        da = DualArray(x_val, t)
        result = np.arcsin(da)
        expected_tangent = t / np.sqrt(1 - x_val**2)
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    def test_tangent_finite_diff(self):
        x = np.array(0.3)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.arcsin, x, v)
        da = DualArray(x, v)
        result = np.arcsin(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestArccos:
    @pytest.mark.parametrize("x_val", ARCSIN_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.arccos(da)
        np.testing.assert_allclose(result.primal, np.arccos(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", ARCSIN_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val)
        da = DualArray(x_val, t)
        result = np.arccos(da)
        expected_tangent = -t / np.sqrt(1 - x_val**2)
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    def test_tangent_finite_diff(self):
        x = np.array(0.3)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.arccos, x, v)
        da = DualArray(x, v)
        result = np.arccos(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestArctan:
    @pytest.mark.parametrize(
        "x_val",
        [
            pytest.param(np.array(0.5), id="scalar_0.5"),
            pytest.param(np.array(2.0), id="scalar_2.0"),
            pytest.param(np.array([0.1, 1.0, -2.0]), id="vector"),
        ],
    )
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.arctan(da)
        np.testing.assert_allclose(result.primal, np.arctan(x_val), rtol=1e-7)

    def test_tangent_analytical(self):
        x_val = np.array(1.5)
        t = np.array(1.0)
        da = DualArray(x_val, t)
        result = np.arctan(da)
        expected_tangent = t / (1 + x_val**2)
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    def test_tangent_finite_diff(self):
        x = np.array(1.5)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.arctan, x, v)
        da = DualArray(x, v)
        result = np.arctan(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestArctan2:
    def test_primal(self):
        y = np.array(1.0)
        x = np.array(1.0)
        da_y = DualArray(y, np.array(1.0))
        da_x = DualArray(x, np.array(0.0))
        result = np.arctan2(da_y, da_x)
        np.testing.assert_allclose(result.primal, np.arctan2(y, x), rtol=1e-7)

    def test_tangent_analytical(self):
        y_val = np.array(3.0)
        x_val = np.array(4.0)
        dy = np.array(1.0)
        dx = np.array(0.0)
        da_y = DualArray(y_val, dy)
        da_x = DualArray(x_val, dx)
        result = np.arctan2(da_y, da_x)
        denom = x_val**2 + y_val**2
        expected_tangent = (x_val * dy - y_val * dx) / denom
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    def test_finite_diff_y(self):
        y = np.array(3.0)
        x = np.array(4.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(lambda y_: np.arctan2(y_, x), y, v)
        result = np.arctan2(DualArray(y, v), DualArray(x, np.array(0.0)))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_vector(self):
        y = DualArray(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
        x = DualArray(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        result = np.arctan2(y, x)
        np.testing.assert_allclose(
            result.primal, np.arctan2([1.0, -1.0], [1.0, 1.0]), rtol=1e-7
        )


class TestExp:
    @pytest.mark.parametrize(
        "x_val",
        [
            pytest.param(np.array(1.0), id="scalar"),
            pytest.param(np.array([0.5, 1.0, 2.0]), id="vector"),
        ],
    )
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.exp(da)
        np.testing.assert_allclose(result.primal, np.exp(x_val), rtol=1e-7)

    @pytest.mark.parametrize(
        "x_val",
        [
            pytest.param(np.array(1.0), id="scalar"),
            pytest.param(np.array([0.5, 1.0, 2.0]), id="vector"),
        ],
    )
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val) * 3.0
        da = DualArray(x_val, t)
        result = np.exp(da)
        expected_tangent = np.exp(x_val) * t
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(2.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.exp, x, v)
        da = DualArray(x, v)
        result = np.exp(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestLog:
    @pytest.mark.parametrize(
        "x_val",
        [
            pytest.param(np.array(2.0), id="scalar"),
            pytest.param(np.array([0.5, 1.0, 3.0]), id="vector"),
        ],
    )
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.log(da)
        np.testing.assert_allclose(result.primal, np.log(x_val), rtol=1e-7)

    @pytest.mark.parametrize(
        "x_val",
        [
            pytest.param(np.array(2.0), id="scalar"),
            pytest.param(np.array([0.5, 1.0, 3.0]), id="vector"),
        ],
    )
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val) * 2.0
        da = DualArray(x_val, t)
        result = np.log(da)
        expected_tangent = t / x_val
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(2.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.log, x, v)
        da = DualArray(x, v)
        result = np.log(da)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestSqrt:
    def test_sqrt_primal_and_tangent(self):
        x = np.array(4.0)
        t = np.array(1.0)
        da = DualArray(x, t)
        result = np.sqrt(da)
        assert_dual_close(result, np.array(2.0), np.array(0.25))


class TestExp2:
    def test_primal(self):
        x = np.array(3.0)
        da = DualArray(x, np.ones_like(x))
        result = np.exp2(da)
        np.testing.assert_allclose(result.primal, np.exp2(x), rtol=1e-7)

    def test_tangent_analytical(self):
        x = np.array(3.0)
        t = np.array(2.0)
        da = DualArray(x, t)
        result = np.exp2(da)
        expected = np.log(2) * np.exp2(x) * t
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(2.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.exp2, x, v)
        result = np.exp2(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestExpm1:
    def test_primal(self):
        x = np.array(0.5)
        da = DualArray(x, np.ones_like(x))
        result = np.expm1(da)
        np.testing.assert_allclose(result.primal, np.expm1(x), rtol=1e-7)

    def test_tangent_analytical(self):
        x = np.array(0.5)
        t = np.array(1.0)
        da = DualArray(x, t)
        result = np.expm1(da)
        expected = np.exp(x) * t
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(0.5)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.expm1, x, v)
        result = np.expm1(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestLog2:
    def test_primal(self):
        x = np.array(4.0)
        da = DualArray(x, np.ones_like(x))
        result = np.log2(da)
        np.testing.assert_allclose(result.primal, np.log2(x), rtol=1e-7)

    def test_tangent_analytical(self):
        x = np.array(4.0)
        t = np.array(1.0)
        da = DualArray(x, t)
        result = np.log2(da)
        expected = t / (x * np.log(2))
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(4.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.log2, x, v)
        result = np.log2(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestLog10:
    def test_primal(self):
        x = np.array(100.0)
        da = DualArray(x, np.ones_like(x))
        result = np.log10(da)
        np.testing.assert_allclose(result.primal, np.log10(x), rtol=1e-7)

    def test_tangent_analytical(self):
        x = np.array(100.0)
        t = np.array(1.0)
        da = DualArray(x, t)
        result = np.log10(da)
        expected = t / (x * np.log(10))
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(10.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.log10, x, v)
        result = np.log10(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestLog1p:
    def test_primal(self):
        x = np.array(0.5)
        da = DualArray(x, np.ones_like(x))
        result = np.log1p(da)
        np.testing.assert_allclose(result.primal, np.log1p(x), rtol=1e-7)

    def test_tangent_analytical(self):
        x = np.array(0.5)
        t = np.array(1.0)
        da = DualArray(x, t)
        result = np.log1p(da)
        expected = t / (1 + x)
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(0.5)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.log1p, x, v)
        result = np.log1p(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


HYPER_SCALAR_VALS = [
    pytest.param(np.array(0.5), id="scalar_0.5"),
    pytest.param(np.array(1.0), id="scalar_1.0"),
]

HYPER_VECTOR_VALS = [
    pytest.param(np.array([0.3, 0.7, 1.2]), id="vector"),
]


class TestSinh:
    @pytest.mark.parametrize("x_val", HYPER_SCALAR_VALS + HYPER_VECTOR_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.sinh(da)
        np.testing.assert_allclose(result.primal, np.sinh(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", HYPER_SCALAR_VALS + HYPER_VECTOR_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val) * 2.0
        da = DualArray(x_val, t)
        result = np.sinh(da)
        np.testing.assert_allclose(result.tangent, np.cosh(x_val) * t, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(1.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.sinh, x, v)
        result = np.sinh(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestCosh:
    @pytest.mark.parametrize("x_val", HYPER_SCALAR_VALS + HYPER_VECTOR_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.cosh(da)
        np.testing.assert_allclose(result.primal, np.cosh(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", HYPER_SCALAR_VALS + HYPER_VECTOR_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val) * 2.0
        da = DualArray(x_val, t)
        result = np.cosh(da)
        np.testing.assert_allclose(result.tangent, np.sinh(x_val) * t, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(1.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.cosh, x, v)
        result = np.cosh(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestTanh:
    @pytest.mark.parametrize("x_val", HYPER_SCALAR_VALS + HYPER_VECTOR_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.tanh(da)
        np.testing.assert_allclose(result.primal, np.tanh(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", HYPER_SCALAR_VALS + HYPER_VECTOR_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val)
        da = DualArray(x_val, t)
        result = np.tanh(da)
        expected = (1 - np.tanh(x_val) ** 2) * t
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(0.5)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.tanh, x, v)
        result = np.tanh(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


ARCSINH_VALS = [
    pytest.param(np.array(0.5), id="scalar_0.5"),
    pytest.param(np.array(2.0), id="scalar_2.0"),
    pytest.param(np.array([0.3, 1.0, -0.5]), id="vector"),
]


class TestArcsinh:
    @pytest.mark.parametrize("x_val", ARCSINH_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.arcsinh(da)
        np.testing.assert_allclose(result.primal, np.arcsinh(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", ARCSINH_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val)
        da = DualArray(x_val, t)
        result = np.arcsinh(da)
        expected = t / np.sqrt(x_val**2 + 1)
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(1.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.arcsinh, x, v)
        result = np.arcsinh(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


ARCCOSH_VALS = [
    pytest.param(np.array(1.5), id="scalar_1.5"),
    pytest.param(np.array(3.0), id="scalar_3.0"),
    pytest.param(np.array([1.2, 2.0, 4.0]), id="vector"),
]


class TestArccosh:
    @pytest.mark.parametrize("x_val", ARCCOSH_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.arccosh(da)
        np.testing.assert_allclose(result.primal, np.arccosh(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", ARCCOSH_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val)
        da = DualArray(x_val, t)
        result = np.arccosh(da)
        expected = t / np.sqrt(x_val**2 - 1)
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(2.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.arccosh, x, v)
        result = np.arccosh(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


ARCTANH_VALS = [
    pytest.param(np.array(0.3), id="scalar_0.3"),
    pytest.param(np.array(-0.5), id="scalar_-0.5"),
    pytest.param(np.array([0.1, 0.4, -0.3]), id="vector"),
]


class TestArctanh:
    @pytest.mark.parametrize("x_val", ARCTANH_VALS)
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.arctanh(da)
        np.testing.assert_allclose(result.primal, np.arctanh(x_val), rtol=1e-7)

    @pytest.mark.parametrize("x_val", ARCTANH_VALS)
    def test_tangent_analytical(self, x_val):
        t = np.ones_like(x_val)
        da = DualArray(x_val, t)
        result = np.arctanh(da)
        expected = t / (1 - x_val**2)
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(0.3)
        v = np.array(1.0)
        fd = finite_diff_jvp(np.arctanh, x, v)
        result = np.arctanh(DualArray(x, v))
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestGreater:
    def test_scalar(self):
        a = DualArray(np.array(3.0), np.array(1.0))
        b = DualArray(np.array(2.0), np.array(1.0))
        result = np.greater(a, b)
        assert result == True  # noqa: E712
        assert not isinstance(result, DualArray)

    def test_vector(self):
        a = DualArray(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
        b = DualArray(np.array([2.0, 2.0]), np.array([1.0, 1.0]))
        result = np.greater(a, b)
        np.testing.assert_array_equal(result, [False, True])


class TestGreaterEqual:
    def test_scalar(self):
        a = DualArray(np.array(2.0), np.array(1.0))
        b = DualArray(np.array(2.0), np.array(1.0))
        result = np.greater_equal(a, b)
        assert result == True  # noqa: E712

    def test_vector(self):
        a = DualArray(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
        b = DualArray(np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        result = np.greater_equal(a, b)
        np.testing.assert_array_equal(result, [False, True, True])


class TestLess:
    def test_scalar(self):
        a = DualArray(np.array(1.0), np.array(1.0))
        b = DualArray(np.array(2.0), np.array(1.0))
        result = np.less(a, b)
        assert result == True  # noqa: E712

    def test_vector(self):
        a = DualArray(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
        b = DualArray(np.array([2.0, 2.0]), np.array([1.0, 1.0]))
        result = np.less(a, b)
        np.testing.assert_array_equal(result, [True, False])


class TestLessEqual:
    def test_vector(self):
        a = DualArray(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
        b = DualArray(np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        result = np.less_equal(a, b)
        np.testing.assert_array_equal(result, [True, True, False])


class TestEqual:
    def test_vector(self):
        a = DualArray(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
        b = DualArray(np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        result = np.equal(a, b)
        np.testing.assert_array_equal(result, [False, True, False])


class TestNotEqual:
    def test_vector(self):
        a = DualArray(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
        b = DualArray(np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        result = np.not_equal(a, b)
        np.testing.assert_array_equal(result, [True, False, True])


class TestOperatorSyntax:
    def test_gt(self):
        a = DualArray(np.array(3.0), np.array(1.0))
        b = DualArray(np.array(2.0), np.array(1.0))
        assert (a > b) == True  # noqa: E712

    def test_lt(self):
        a = DualArray(np.array(1.0), np.array(1.0))
        b = DualArray(np.array(2.0), np.array(1.0))
        assert (a < b) == True  # noqa: E712

    def test_ge(self):
        a = DualArray(np.array(2.0), np.array(1.0))
        b = DualArray(np.array(2.0), np.array(1.0))
        assert (a >= b) == True  # noqa: E712

    def test_le(self):
        a = DualArray(np.array(2.0), np.array(1.0))
        b = DualArray(np.array(2.0), np.array(1.0))
        assert (a <= b) == True  # noqa: E712

    def test_eq_operator(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([1.0, 3.0]), np.array([0.5, 0.5]))
        result = a == b
        np.testing.assert_array_equal(result, np.array([True, False]))

    def test_ne_operator(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([1.0, 3.0]), np.array([0.5, 0.5]))
        result = a != b
        np.testing.assert_array_equal(result, np.array([False, True]))


class TestEdgeCases:
    def test_absolute_at_zero(self):
        da = DualArray(np.array(0.0), np.array(1.0))
        result = np.absolute(da)
        assert np.isfinite(result.tangent)
        assert_tangent_finite(result)

    def test_cbrt_near_zero(self):
        da = DualArray(np.array(1e-15), np.array(1.0))
        result = np.cbrt(da)
        assert np.isfinite(result.primal)

    def test_reciprocal_near_zero(self):
        da = DualArray(np.array(1e-15), np.array(1.0))
        result = np.reciprocal(da)
        assert not np.isnan(result.tangent)

    def test_power_zero_base_integer_exp(self):
        da = DualArray(np.array(0.0), np.array(1.0))
        result = da**3
        np.testing.assert_allclose(result.primal, 0.0)
        np.testing.assert_allclose(result.tangent, 0.0)

    def test_dual_array_int_primal(self):
        da = DualArray(np.array([1, 2, 3]))
        result = da + da
        assert isinstance(result, DualArray)

    def test_broadcast_tangent_through_add(self):
        da1 = DualArray(np.ones((2, 2)), np.ones(2))
        da2 = DualArray(np.ones((2, 2)))
        result = da1 + da2
        assert isinstance(result, DualArray)
        assert result.shape == (2, 2)

    def test_broadcast_tangent_through_multiply(self):
        da1 = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.1, 0.2]))
        da2 = DualArray(np.array([[5.0, 6.0], [7.0, 8.0]]))
        result = da1 * da2
        assert isinstance(result, DualArray)
        assert result.shape == (2, 2)


class TestPositive:
    def test_primal(self, x_val):
        da = DualArray(x_val, np.ones_like(x_val))
        result = np.positive(da)
        np.testing.assert_allclose(result.primal, np.positive(x_val))

    def test_tangent(self, x_val):
        t = np.ones_like(x_val) * 0.7
        da = DualArray(x_val, t)
        result = np.positive(da)
        np.testing.assert_allclose(result.tangent, t)


class TestFloatPower:
    def test_primal(self):
        x = np.array([2.0, 3.0])
        da = DualArray(x, np.ones_like(x))
        result = np.float_power(da, 3.0)
        np.testing.assert_allclose(result.primal, np.float_power(x, 3.0))

    def test_tangent(self):
        x = np.array([2.0, 3.0])
        t = np.array([1.0, 1.0])
        da = DualArray(x, t)
        result = np.float_power(da, 3.0)
        np.testing.assert_allclose(result.tangent, 3.0 * x**2 * t, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array([2.0, 3.0])
        v = np.array([1.0, 0.5])
        fd = finite_diff_jvp(lambda z: np.float_power(z, 2.5), x, v)
        da = DualArray(x, v)
        result = np.float_power(da, 2.5)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_dual_base_dual_exp(self):
        x = np.array(2.0)
        n = np.array(3.0)
        dx = DualArray(x, np.array(1.0))
        dn = DualArray(n, np.array(0.5))
        result = np.float_power(dx, dn)
        np.testing.assert_allclose(result.primal, 8.0)

    def test_scalar_base_dual_exp(self):
        x = 2.0
        n = np.array(3.0)
        dn = DualArray(n, np.array(1.0))
        result = np.float_power(x, dn)
        np.testing.assert_allclose(result.primal, 8.0)
        np.testing.assert_allclose(result.tangent, 8.0 * np.log(2.0), rtol=1e-7)


class TestFabs:
    def test_primal(self):
        x = np.array([-3.0, 2.0, -1.5])
        da = DualArray(x, np.ones_like(x))
        result = np.fabs(da)
        np.testing.assert_allclose(result.primal, np.fabs(x))

    def test_tangent(self):
        x = np.array([-3.0, 2.0, -1.5])
        t = np.array([1.0, 0.5, 2.0])
        da = DualArray(x, t)
        result = np.fabs(da)
        np.testing.assert_allclose(result.tangent, np.sign(x) * t)


class TestRemainder:
    def test_primal(self):
        x = np.array([7.0, 10.0])
        y = np.array([3.0, 4.0])
        dx = DualArray(x, np.ones_like(x))
        dy = DualArray(y, np.zeros_like(y))
        result = np.remainder(dx, dy)
        np.testing.assert_allclose(result.primal, np.remainder(x, y))

    def test_tangent_x(self):
        x = np.array([7.0, 10.0])
        y = np.array([3.0, 4.0])
        dx = DualArray(x, np.ones_like(x))
        dy = DualArray(y, np.zeros_like(y))
        result = np.remainder(dx, dy)
        np.testing.assert_allclose(result.tangent, np.ones_like(x), rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(7.5)
        v = np.array(1.0)
        fd = finite_diff_jvp(lambda z: np.remainder(z, 3.0), x, v)
        dx = DualArray(x, v)
        dy = DualArray(np.array(3.0), np.array(0.0))
        result = np.remainder(dx, dy)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)


class TestFmod:
    def test_primal(self):
        x = np.array([7.0, -10.0])
        y = np.array([3.0, 4.0])
        dx = DualArray(x, np.ones_like(x))
        dy = DualArray(y, np.zeros_like(y))
        result = np.fmod(dx, dy)
        np.testing.assert_allclose(result.primal, np.fmod(x, y))

    def test_tangent_x(self):
        x = np.array([7.0, -10.0])
        y = np.array([3.0, 4.0])
        dx = DualArray(x, np.ones_like(x))
        dy = DualArray(y, np.zeros_like(y))
        result = np.fmod(dx, dy)
        np.testing.assert_allclose(result.tangent, np.ones_like(x), rtol=1e-7)


class TestCopysign:
    def test_primal(self):
        x = np.array([3.0, -2.0])
        y = np.array([-1.0, 5.0])
        dx = DualArray(x, np.ones_like(x))
        dy = DualArray(y, np.zeros_like(y))
        result = np.copysign(dx, dy)
        np.testing.assert_allclose(result.primal, np.copysign(x, y))

    def test_tangent(self):
        x = np.array([3.0, -2.0])
        y = np.array([-1.0, 5.0])
        t = np.array([1.0, 1.0])
        dx = DualArray(x, t)
        dy = DualArray(y, np.zeros_like(y))
        result = np.copysign(dx, dy)
        expected_tangent = np.sign(y) * t
        np.testing.assert_allclose(result.tangent, expected_tangent, rtol=1e-7)


class TestHypot:
    def test_primal(self):
        x = np.array([3.0, 5.0])
        y = np.array([4.0, 12.0])
        dx = DualArray(x, np.ones_like(x))
        dy = DualArray(y, np.zeros_like(y))
        result = np.hypot(dx, dy)
        np.testing.assert_allclose(result.primal, np.hypot(x, y))

    def test_tangent(self):
        x = np.array(3.0)
        y = np.array(4.0)
        dx = DualArray(x, np.array(1.0))
        dy = DualArray(y, np.array(0.0))
        result = np.hypot(dx, dy)
        np.testing.assert_allclose(result.tangent, 3.0 / 5.0, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(3.0)
        y = np.array(4.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(lambda z: np.hypot(z, y), x, v)
        dx = DualArray(x, v)
        dy = DualArray(y, np.array(0.0))
        result = np.hypot(dx, dy)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_at_zero(self):
        dx = DualArray(np.array(0.0), np.array(1.0))
        dy = DualArray(np.array(0.0), np.array(1.0))
        result = np.hypot(dx, dy)
        assert np.isfinite(result.tangent)


class TestDegrees:
    def test_primal(self):
        x = np.array([0.0, np.pi / 2, np.pi])
        da = DualArray(x, np.ones_like(x))
        result = np.degrees(da)
        np.testing.assert_allclose(result.primal, np.degrees(x))

    def test_tangent(self):
        x = np.array(np.pi)
        t = np.array(1.0)
        da = DualArray(x, t)
        result = np.degrees(da)
        np.testing.assert_allclose(result.tangent, 180.0 / np.pi, rtol=1e-7)


class TestRad2deg:
    def test_primal(self):
        x = np.array(np.pi / 4)
        da = DualArray(x, np.array(1.0))
        result = np.rad2deg(da)
        np.testing.assert_allclose(result.primal, np.rad2deg(x))

    def test_tangent(self):
        x = np.array(1.0)
        da = DualArray(x, np.array(1.0))
        result = np.rad2deg(da)
        np.testing.assert_allclose(result.tangent, 180.0 / np.pi, rtol=1e-7)


class TestRadians:
    def test_primal(self):
        x = np.array([0.0, 90.0, 180.0])
        da = DualArray(x, np.ones_like(x))
        result = np.radians(da)
        np.testing.assert_allclose(result.primal, np.radians(x))

    def test_tangent(self):
        x = np.array(180.0)
        da = DualArray(x, np.array(1.0))
        result = np.radians(da)
        np.testing.assert_allclose(result.tangent, np.pi / 180.0, rtol=1e-7)


class TestDeg2rad:
    def test_primal(self):
        x = np.array(45.0)
        da = DualArray(x, np.array(1.0))
        result = np.deg2rad(da)
        np.testing.assert_allclose(result.primal, np.deg2rad(x))

    def test_tangent(self):
        x = np.array(90.0)
        da = DualArray(x, np.array(1.0))
        result = np.deg2rad(da)
        np.testing.assert_allclose(result.tangent, np.pi / 180.0, rtol=1e-7)


class TestLogaddexp:
    def test_primal(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 0.5])
        dx = DualArray(x, np.ones_like(x))
        dy = DualArray(y, np.zeros_like(y))
        result = np.logaddexp(dx, dy)
        np.testing.assert_allclose(result.primal, np.logaddexp(x, y))

    def test_tangent(self):
        x = np.array(1.0)
        y = np.array(2.0)
        dx = DualArray(x, np.array(1.0))
        dy = DualArray(y, np.array(0.0))
        result = np.logaddexp(dx, dy)
        expected = np.exp(x) / (np.exp(x) + np.exp(y))
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(1.0)
        y = np.array(2.0)
        v = np.array(1.0)
        fd = finite_diff_jvp(lambda z: np.logaddexp(z, y), x, v)
        dx = DualArray(x, v)
        dy = DualArray(y, np.array(0.0))
        result = np.logaddexp(dx, dy)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)

    def test_large_values(self):
        x = np.array(100.0)
        y = np.array(0.0)
        dx = DualArray(x, np.array(1.0))
        dy = DualArray(y, np.array(0.0))
        result = np.logaddexp(dx, dy)
        assert_tangent_finite(result)
        np.testing.assert_allclose(result.tangent, 1.0, rtol=1e-5)


class TestLogaddexp2:
    def test_primal(self):
        x = np.array([1.0, 3.0])
        y = np.array([2.0, 1.0])
        dx = DualArray(x, np.ones_like(x))
        dy = DualArray(y, np.zeros_like(y))
        result = np.logaddexp2(dx, dy)
        np.testing.assert_allclose(result.primal, np.logaddexp2(x, y))

    def test_tangent(self):
        x = np.array(1.0)
        y = np.array(2.0)
        dx = DualArray(x, np.array(1.0))
        dy = DualArray(y, np.array(0.0))
        result = np.logaddexp2(dx, dy)
        expected = np.exp2(x) / (np.exp2(x) + np.exp2(y))
        np.testing.assert_allclose(result.tangent, expected, rtol=1e-7)

    def test_finite_diff(self):
        x = np.array(1.5)
        y = np.array(2.5)
        v = np.array(1.0)
        fd = finite_diff_jvp(lambda z: np.logaddexp2(z, y), x, v)
        dx = DualArray(x, v)
        dy = DualArray(y, np.array(0.0))
        result = np.logaddexp2(dx, dy)
        np.testing.assert_allclose(result.tangent, fd, rtol=1e-5)
