import numpy as np
import pytest
from dualpy.differentiation import jacobian

from tests.conftest import finite_diff_jvp

pytestmark = pytest.mark.integration


class TestJacobian:
    def test_scalar_to_scalar(self):
        """f(x) = x^3, J = 3x^2.  At x=2, J=12."""

        def f(x):
            return x**3

        jac = jacobian(f)(np.array(2.0))
        np.testing.assert_allclose(jac, np.array(12.0), rtol=1e-7)

    def test_linear_map(self):
        """f(x) = [2*x0, 3*x1, x0*x1].  Jacobian at [1, 0]."""

        def f(x):
            return np.stack([2 * x[0], 3 * x[1], x[0] * x[1]])

        jac = jacobian(f)(np.array([1.0, 0.0]))
        expected = np.array([[2.0, 0.0], [0.0, 3.0], [0.0, 1.0]])
        np.testing.assert_allclose(jac, expected, rtol=1e-7)

    def test_jvp_with_direction_vector(self):
        """f(x) = x^2, J@v at x=3 with v=1 should give 6."""

        def f(x):
            return x**2

        v = np.array(1.0)
        jvp = jacobian(f, v=v)(np.array(3.0))
        np.testing.assert_allclose(jvp, np.array(6.0), rtol=1e-7)

    def test_jacobian_vs_finite_diff(self):
        """Cross-check Jacobian against finite differences."""

        def f(x):
            return np.stack([x[0] ** 2 + x[1], x[0] * x[1]])

        x0 = np.array([2.0, 3.0])
        jac = jacobian(f)(x0)

        def f_plain(x):
            return np.array([x[0] ** 2 + x[1], x[0] * x[1]])

        for i in range(2):
            e_i = np.zeros(2)
            e_i[i] = 1.0
            fd_col = finite_diff_jvp(f_plain, x0, e_i)
            np.testing.assert_allclose(jac[:, i], fd_col, rtol=1e-5)

    def test_accepts_matrix_input(self):
        """Matrix input should be accepted after n-D support."""

        def f(X):
            return X[0, 0] ** 2

        J = jacobian(f)(np.array([[3.0, 0.0], [0.0, 0.0]]))
        assert J.shape == (2, 2)
        expected = np.array([[6.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(J, expected, atol=1e-12)

    def test_jvp_shape_mismatch(self):
        def f(x):
            return x**2

        v = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="shape mismatch"):
            jacobian(f, v=v)(np.array(3.0))


class TestDerivative:
    def test_polynomial(self):
        from dualpy.differentiation import derivative

        def f(x):
            return x**2

        df = derivative(f)
        np.testing.assert_allclose(df(2.0), 4.0, rtol=1e-7)

    def test_sin(self):
        from dualpy.differentiation import derivative

        df = derivative(np.sin)
        np.testing.assert_allclose(df(1.0), np.cos(1.0), rtol=1e-7)

    def test_composition(self):
        from dualpy.differentiation import derivative

        def f(x):
            return np.exp(np.sin(x))

        df = derivative(f)
        x = np.array(1.0)
        expected = np.exp(np.sin(x)) * np.cos(x)
        np.testing.assert_allclose(df(x), expected, rtol=1e-7)

    def test_rejects_vector_input(self):
        from dualpy.differentiation import derivative

        def f(x):
            return x**2

        with pytest.raises(ValueError, match="scalar input"):
            derivative(f)(np.array([1.0, 2.0]))

    def test_rejects_vector_valued_function(self):
        from dualpy.differentiation import derivative

        def f(x):
            return np.stack([x, x**2])

        with pytest.raises(ValueError, match="scalar-valued function"):
            derivative(f)(1.0)


class TestGradient:
    def test_quadratic(self):
        from dualpy.differentiation import gradient

        def g(x):
            return x[0] ** 2 + x[1] ** 2

        grad_g = gradient(g)
        x0 = np.array([3.0, 4.0])
        result = grad_g(x0)
        np.testing.assert_allclose(result, np.array([6.0, 8.0]), rtol=1e-7)

    def test_with_direction(self):
        from dualpy.differentiation import gradient

        def g(x):
            return x[0] ** 2 + x[1] ** 2

        v = np.array([1.0, 0.0])
        directional = gradient(g, v=v)
        x0 = np.array([3.0, 4.0])
        result = directional(x0)
        np.testing.assert_allclose(result, 6.0, rtol=1e-7)

    def test_rejects_vector_valued_function(self):
        from dualpy.differentiation import gradient

        def f(x):
            return np.stack([x[0] ** 2, x[1] ** 2])

        with pytest.raises(ValueError, match="scalar-valued function"):
            gradient(f)(np.array([1.0, 2.0]))

    def test_rejects_vector_valued_with_direction(self):
        from dualpy.differentiation import gradient

        def f(x):
            return np.stack([x[0], x[1]])

        v = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="scalar-valued function"):
            gradient(f, v=v)(np.array([1.0, 2.0]))


class TestComposition:
    def test_second_derivative_polynomial(self):
        """f(x) = x^3, f''(x) = 6x. At x=2, f''=12."""
        from dualpy.differentiation import derivative

        def f(x):
            return x**3

        ddf = derivative(derivative(f))
        np.testing.assert_allclose(ddf(2.0), 12.0, atol=1e-12)

    def test_second_derivative_sin(self):
        """f(x) = sin(x), f''(x) = -sin(x)."""
        from dualpy.differentiation import derivative

        ddf = derivative(derivative(np.sin))
        np.testing.assert_allclose(ddf(1.0), -np.sin(1.0), atol=1e-12)

    def test_third_derivative(self):
        """f(x) = x^4, f'''(x) = 24x. At x=1, f'''=24."""
        from dualpy.differentiation import derivative

        def f(x):
            return x**4

        dddf = derivative(derivative(derivative(f)))
        np.testing.assert_allclose(dddf(1.0), 24.0, atol=1e-12)

    def test_jacobian_of_gradient_is_hessian(self):
        """jacobian(gradient(f)) should give exact Hessian."""
        from dualpy.differentiation import gradient

        def f(x):
            return x[0] ** 2 + 3 * x[1] ** 2 + x[0] * x[1]

        jac_grad = jacobian(gradient(f))
        H = jac_grad(np.array([1.0, 2.0]))
        expected = np.array([[2.0, 1.0], [1.0, 6.0]])
        np.testing.assert_allclose(H, expected, atol=1e-12)

    def test_laplacian_exact(self):
        """Laplacian should now be exact (no finite differences)."""
        from dualpy.differentiation import laplacian

        def f(x):
            return x[0] ** 3 + x[1] ** 3

        result = laplacian(f)(np.array([2.0, 3.0]))
        expected = 6 * 2.0 + 6 * 3.0
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestHessian:
    def test_quadratic_form(self):
        from dualpy.differentiation import hessian

        A = np.array([[2.0, 1.0], [1.0, 3.0]])

        def f(x):
            return x @ A @ x

        H = hessian(f)(np.array([1.0, 1.0]))
        expected = A + A.T
        np.testing.assert_allclose(H, expected, atol=1e-12)


class TestObjectArrayWarning:
    def test_np_array_emits_warning(self):
        def f(x):
            return np.array([x[0], x[1]])

        with pytest.warns(UserWarning, match="np.stack"):
            jacobian(f)(np.array([1.0, 2.0]))


class TestCurl:
    def test_curl_of_gradient_is_zero(self):
        from dualpy.differentiation import curl

        def grad_field(x):
            return np.stack([2 * x[0], 2 * x[1], 2 * x[2]])

        result = curl(grad_field)(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, np.zeros(3), atol=1e-7)


class TestDivergence:
    def test_identity_field(self):
        from dualpy.differentiation import divergence

        def F(x):
            return x

        result = divergence(F)(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, 3.0, rtol=1e-7)


class TestLaplacian:
    def test_quadratic(self):
        from dualpy.differentiation import laplacian

        def f(x):
            return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

        result = laplacian(f)(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, 6.0, atol=1e-12)


class TestComplex:
    def test_derivative_polynomial(self):
        """f(z) = z^2, f'(z) = 2z. At z=1+2j, f'=2+4j."""
        from dualpy.differentiation import derivative

        def f(z):
            return z**2

        result = derivative(f)(1 + 2j)
        np.testing.assert_allclose(result, 2 + 4j, atol=1e-12)

    def test_derivative_exp(self):
        """f(z) = exp(z), f'(z) = exp(z). At z=j*pi, f'=exp(j*pi)=-1."""
        from dualpy.differentiation import derivative

        result = derivative(np.exp)(1j * np.pi)
        np.testing.assert_allclose(result, np.exp(1j * np.pi), atol=1e-12)

    def test_derivative_sin(self):
        """f(z) = sin(z), f'(z) = cos(z) for complex z."""
        from dualpy.differentiation import derivative

        z = 1.0 + 1.0j
        result = derivative(np.sin)(z)
        np.testing.assert_allclose(result, np.cos(z), atol=1e-12)

    def test_gradient_complex(self):
        """Gradient of a scalar-valued function of complex vector."""
        from dualpy.differentiation import gradient

        def f(z):
            return z[0] ** 2 + z[1] ** 2

        z0 = np.array([1 + 1j, 2 + 0j])
        result = gradient(f)(z0)
        expected = np.array([2 * (1 + 1j), 2 * (2 + 0j)])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_jacobian_complex(self):
        """Jacobian of a vector-valued holomorphic function."""

        def f(z):
            return np.stack([z[0] ** 2, z[0] * z[1]])

        z0 = np.array([1 + 1j, 2 + 0j])
        J = jacobian(f)(z0)
        expected = np.array(
            [
                [2 * (1 + 1j), 0 + 0j],
                [2 + 0j, 1 + 1j],
            ]
        )
        np.testing.assert_allclose(J, expected, atol=1e-12)


class TestNDimensional:
    def test_jacobian_matrix_input(self):
        """Jacobian of a scalar-valued function of a matrix."""

        def f(X):
            return X[0, 0] ** 2 + X[0, 1] * X[1, 0]

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        J = jacobian(f)(X)
        assert J.shape == (2, 2)
        expected = np.array([[2.0, 3.0], [2.0, 0.0]])
        np.testing.assert_allclose(J, expected, atol=1e-12)

    def test_gradient_matrix_input(self):
        """Gradient of a scalar-valued function of a matrix."""
        from dualpy.differentiation import gradient

        def f(X):
            return X[0, 0] ** 2 + X[0, 1] * X[1, 0]

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad = gradient(f)(X)
        assert grad.shape == (2, 2)
        expected = np.array([[2.0, 3.0], [2.0, 0.0]])
        np.testing.assert_allclose(grad, expected, atol=1e-12)

    def test_hessian_matrix_input(self):
        """Hessian of a scalar-valued function of a matrix."""
        from dualpy.differentiation import hessian

        def f(X):
            return X[0, 0] ** 2 + X[0, 1] * X[1, 0]

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        H = hessian(f)(X)
        assert H.shape == (2, 2, 2, 2)
        H_flat = H.reshape(4, 4)
        expected_flat = np.zeros((4, 4))
        expected_flat[0, 0] = 2.0
        expected_flat[1, 2] = 1.0
        expected_flat[2, 1] = 1.0
        np.testing.assert_allclose(H_flat, expected_flat, atol=1e-12)

    def test_laplacian_matrix_input(self):
        """Laplacian of a scalar-valued function of a matrix."""
        from dualpy.differentiation import laplacian

        def f(X):
            return X[0, 0] ** 2 + X[1, 1] ** 2

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = laplacian(f)(X)
        np.testing.assert_allclose(result, 4.0, atol=1e-12)

    def test_jacobian_matrix_to_vector(self):
        """Jacobian of a vector-valued function of a matrix."""

        def f(X):
            return np.stack([X[0, 0] + X[1, 1], X[0, 1] * X[1, 0]])

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        J = jacobian(f)(X)
        assert J.shape == (2, 2, 2)
        expected = np.zeros((2, 2, 2))
        expected[0, 0, 0] = 1.0
        expected[0, 1, 1] = 1.0
        expected[1, 0, 1] = 3.0
        expected[1, 1, 0] = 2.0
        np.testing.assert_allclose(J, expected, atol=1e-12)

    def test_broadcasting_bug_fixed(self):
        """Element-wise vector ops that failed with batched identity seed."""

        def f(x):
            return x * x[::-1]

        x = np.array([2.0, 3.0])
        J = jacobian(f)(x)
        expected = np.array([[3.0, 2.0], [3.0, 2.0]])
        np.testing.assert_allclose(J, expected, atol=1e-12)


class TestJVP:
    def test_scalar_function(self):
        """f(x) = x^2, at x=3 with tangent=1: primal=9, tangent=6."""
        from dualpy.differentiation import jvp

        primal, tangent = jvp(lambda x: x**2, np.array(3.0), np.array(1.0))
        np.testing.assert_allclose(primal, 9.0, atol=1e-12)
        np.testing.assert_allclose(tangent, 6.0, atol=1e-12)

    def test_vector_function(self):
        """f(x) = [x0^2, x0*x1], tangent along e0 gives first Jacobian column."""
        from dualpy.differentiation import jvp

        def f(x):
            return np.stack([x[0] ** 2, x[0] * x[1]])

        x = np.array([2.0, 3.0])
        v = np.array([1.0, 0.0])
        primal, tangent = jvp(f, x, v)
        np.testing.assert_allclose(primal, np.array([4.0, 6.0]), atol=1e-12)
        np.testing.assert_allclose(tangent, np.array([4.0, 3.0]), atol=1e-12)

    def test_matches_finite_differences(self):
        from dualpy.differentiation import jvp

        def f(x):
            return np.stack([np.sin(x[0]) * x[1], np.exp(x[0] + x[1])])

        x = np.array([1.0, 2.0])
        v = np.array([0.3, -0.7])
        _, tangent = jvp(f, x, v)

        def f_plain(x):
            return np.array([np.sin(x[0]) * x[1], np.exp(x[0] + x[1])])

        fd = finite_diff_jvp(f_plain, x, v)
        np.testing.assert_allclose(tangent, fd, rtol=1e-5)

    def test_shape_mismatch(self):
        from dualpy.differentiation import jvp

        with pytest.raises(ValueError, match="shape mismatch"):
            jvp(lambda x: x**2, np.array([1.0, 2.0]), np.array([1.0]))

    def test_primal_unchanged(self):
        """Primal output should equal func(x) exactly."""
        from dualpy.differentiation import jvp

        def f(x):
            return np.sin(x)

        x = np.array(1.5)
        primal, _ = jvp(f, x, np.array(1.0))
        np.testing.assert_allclose(primal, np.sin(1.5), atol=1e-15)


class TestHVP:
    def test_quadratic(self):
        """f(x) = x0^2 + 3*x1^2, H = [[2,0],[0,6]], H@[1,0] = [2,0]."""
        from dualpy.differentiation import hvp

        def f(x):
            return x[0] ** 2 + 3 * x[1] ** 2

        v = np.array([1.0, 0.0])
        result = hvp(f, v)(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, np.array([2.0, 0.0]), atol=1e-12)

    def test_matches_full_hessian(self):
        """hvp(f, v)(x) should equal hessian(f)(x) @ v."""
        from dualpy.differentiation import hessian, hvp

        def f(x):
            return x[0] ** 3 + x[0] * x[1] ** 2 + x[1] ** 3

        x = np.array([2.0, 3.0])
        v = np.array([0.5, -1.0])
        hvp_result = hvp(f, v)(x)
        H = hessian(f)(x)
        expected = H @ v
        np.testing.assert_allclose(hvp_result, expected, atol=1e-12)

    def test_second_direction(self):
        """Verify with a different direction vector."""
        from dualpy.differentiation import hvp

        def f(x):
            return x[0] ** 2 + 3 * x[1] ** 2

        v = np.array([0.0, 1.0])
        result = hvp(f, v)(np.array([5.0, 7.0]))
        np.testing.assert_allclose(result, np.array([0.0, 6.0]), atol=1e-12)

    def test_cross_term(self):
        """f(x) = x0*x1, H = [[0,1],[1,0]], H@[1,1] = [1,1]."""
        from dualpy.differentiation import hvp

        def f(x):
            return x[0] * x[1]

        v = np.array([1.0, 1.0])
        result = hvp(f, v)(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, np.array([1.0, 1.0]), atol=1e-12)


class TestNthDerivative:
    def test_zeroth_is_identity(self):
        """nth_derivative(f, 0) should return f itself."""
        from dualpy.differentiation import nth_derivative

        def f(x):
            return x**2

        f0 = nth_derivative(f, 0)
        np.testing.assert_allclose(f0(3.0), 9.0, atol=1e-12)

    def test_first_matches_derivative(self):
        from dualpy.differentiation import derivative, nth_derivative

        def f(x):
            return np.sin(x)

        x = np.array(1.0)
        np.testing.assert_allclose(
            nth_derivative(f, 1)(x), derivative(f)(x), atol=1e-12
        )

    def test_polynomial_third(self):
        """f(x) = x^4, f'''(x) = 24x. At x=1, f'''=24."""
        from dualpy.differentiation import nth_derivative

        def f(x):
            return x**4

        np.testing.assert_allclose(nth_derivative(f, 3)(1.0), 24.0, atol=1e-12)

    def test_polynomial_fourth(self):
        """f(x) = x^4, f''''(x) = 24 (constant)."""
        from dualpy.differentiation import nth_derivative

        def f(x):
            return x**4

        np.testing.assert_allclose(nth_derivative(f, 4)(2.0), 24.0, atol=1e-12)

    def test_sin_cycle(self):
        """sin^(4)(x) = sin(x) — the 4th derivative cycles back."""
        from dualpy.differentiation import nth_derivative

        x = np.array(0.7)
        np.testing.assert_allclose(nth_derivative(np.sin, 4)(x), np.sin(x), atol=1e-10)

    def test_negative_n_raises(self):
        from dualpy.differentiation import nth_derivative

        with pytest.raises(ValueError, match="non-negative integer"):
            nth_derivative(np.sin, -1)

    def test_non_integer_n_raises(self):
        from dualpy.differentiation import nth_derivative

        with pytest.raises(ValueError, match="non-negative integer"):
            nth_derivative(np.sin, 2.5)
