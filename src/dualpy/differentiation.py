from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .core.dual import DualArray

_ArrayLike = npt.NDArray | DualArray


def _ensure_array(x: Any) -> npt.NDArray | DualArray:
    """Return *x* unchanged if it is a ``DualArray``, otherwise ``np.asarray(x)``."""
    return x if isinstance(x, DualArray) else np.asarray(x)


def _argnums_partial(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    idx: int,
) -> Callable[[_ArrayLike], Any]:
    """Create a closure fixing all positional args except the one at *idx*.

    When the traced argument *x* is a ``DualArray``, every other argument
    is wrapped in ``DualArray(...)`` (zero tangent) so that all arithmetic
    inside *func* stays in dual-number mode.  Re-wrapping an argument that
    is already a ``DualArray`` adds another nesting layer, which keeps
    nesting depths consistent for forward-over-forward second derivatives.
    """

    def partial_func(x: _ArrayLike) -> Any:
        new_args = list(args)
        new_args[idx] = x
        if isinstance(x, DualArray):
            for i, a in enumerate(new_args):
                if i != idx:
                    new_args[i] = DualArray(_ensure_array(a))
        return func(*new_args, **kwargs)

    return partial_func


def _extract_result(
    result: Any,
) -> tuple[npt.NDArray | DualArray, npt.NDArray | DualArray]:
    """Extract primal and tangent from a function result.

    Handles both proper DualArray returns and the fallback where
    ``np.array([da1, da2, ...])`` produces an object array of DualArrays.
    """
    if isinstance(result, DualArray):
        return result.primal, result.tangent
    if isinstance(result, np.ndarray) and result.dtype == object:
        warnings.warn(
            "Function returned a plain NumPy object array instead of a "
            "DualArray. This happens because np.array() does not dispatch "
            "through NumPy's __array_function__ protocol for lists of custom "
            "objects, so tangent information must be recovered element-by-element "
            "(slower and fragile). Use np.stack([...]) instead, which dispatches "
            "correctly and preserves DualArray semantics.",
            stacklevel=3,
        )
        primals = [
            np.asarray(el.primal) if isinstance(el, DualArray) else np.asarray(el)
            for el in result.flat
        ]
        tangents = [
            np.asarray(el.tangent) if isinstance(el, DualArray) else np.float64(0.0)
            for el in result.flat
        ]
        p = np.stack(primals)
        t = np.stack(tangents)
        return (
            p.reshape(result.shape + p.shape[1:]),
            t.reshape(result.shape + t.shape[1:]),
        )
    raise TypeError(
        f"function must return a DualArray (or use np.stack to build "
        f"vector outputs), got {type(result).__name__}"
    )


def _columnwise_jacobian(
    func: Callable[[_ArrayLike], Any],
    x: _ArrayLike,
    v: npt.NDArray | None,
) -> npt.NDArray | DualArray:
    """Per-column Jacobian evaluation for arbitrary-shape inputs.

    Evaluates one perturbation direction at a time via :func:`jvp`,
    which avoids the broadcasting bug inherent in batched identity seeds
    and naturally supports n-dimensional inputs.  Works for both plain
    ndarray and ``DualArray`` (nested differentiation) inputs.
    """
    x_shape = x.shape
    if v is not None:
        return jvp(func, x, v)[1]
    if x.size == 1:
        return jvp(func, x, np.ones(x_shape))[1]
    columns = []
    for idx in np.ndindex(x_shape):
        e = np.zeros(x_shape)
        e[idx] = 1.0
        columns.append(jvp(func, x, e)[1])
    flat = np.stack(columns, axis=-1)
    if x.ndim > 1:
        flat = np.reshape(flat, flat.shape[:-1] + x_shape)
    return flat


_Primals = (
    _ArrayLike | tuple[npt.ArrayLike | DualArray, ...] | list[npt.ArrayLike | DualArray]
)
_Tangents = npt.ArrayLike | DualArray | tuple[npt.ArrayLike, ...] | list[npt.ArrayLike]
_JvpOut = tuple[npt.NDArray | DualArray, npt.NDArray | DualArray]


def jvp(
    func: Callable[..., Any],
    primals: _Primals,
    tangents: _Tangents,
) -> _JvpOut:
    """Evaluate ``func`` and its Jacobian-vector product in one forward pass.

    This is the most fundamental forward-mode AD primitive.  It seeds a
    ``DualArray`` with the given tangent vector and propagates both the
    primal computation and the directional derivative simultaneously.

    When *primals* is a ``DualArray`` (nested differentiation), the
    tangent seed is automatically wrapped so that forward-over-forward
    second derivatives work correctly.

    *primals* may also be a tuple or list of arrays for multi-argument
    functions, in which case *tangents* must be a matching sequence.

    Parameters
    ----------
    func : callable
        A function ``f(x) -> y`` or ``f(x, y, ...) -> y`` where inputs
        and outputs are scalars or arrays of any shape.
    primals : array_like, DualArray, or tuple/list thereof
        The point at which to evaluate *func*.  A tuple or list activates
        multi-argument mode where each element is a separate positional
        argument.  Individual elements may be ``DualArray`` for nested
        differentiation.
    tangents : array_like or tuple/list thereof
        The tangent (seed) vector(s).  Must match the structure and shapes
        of *primals*.

    Returns
    -------
    primal_out : ndarray or DualArray
        The result of ``func(primals)`` (or ``func(*primals)``).
    tangent_out : ndarray or DualArray
        The Jacobian-vector product evaluated at *primals*.

    Raises
    ------
    ValueError
        If *primals* and *tangents* have mismatched shapes.

    Examples
    --------
    Single-argument:

    >>> import numpy as np
    >>> from dualpy import jvp
    >>> f = lambda x: np.stack([x[0]**2, x[0] * x[1]])
    >>> primal_out, tangent_out = jvp(f, np.array([2.0, 3.0]), np.array([1.0, 0.0]))
    >>> primal_out
    array([4., 6.])
    >>> tangent_out
    array([4., 3.])

    Multi-argument:

    >>> g = lambda x, y: x**2 * y
    >>> jvp(g, (np.array(3.0), np.array(2.0)), (np.array(1.0), np.array(0.0)))
    (np.float64(18.0), np.float64(12.0))
    """
    if isinstance(primals, tuple | list):
        dual_args = []
        for p, t in zip(primals, tangents, strict=True):
            p = _ensure_array(p)
            if isinstance(p, DualArray):
                t = DualArray(t, np.zeros_like(t))
            else:
                t = np.asarray(t)
                if p.shape != t.shape:
                    raise ValueError(
                        f"shape mismatch: primal.shape = {p.shape}, "
                        f"tangent.shape = {t.shape}"
                    )
            dual_args.append(DualArray(p, t))
        return _extract_result(func(*dual_args))
    primals = _ensure_array(primals)
    if isinstance(primals, DualArray):
        tangents = DualArray(tangents, np.zeros_like(tangents))
    else:
        tangents = np.asarray(tangents)
        if primals.shape != tangents.shape:
            raise ValueError(
                f"shape mismatch: primals.shape = {primals.shape}, "
                f"tangents.shape = {tangents.shape}"
            )
    return _extract_result(func(DualArray(primals, tangents)))


def jacobian(
    func: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    *,
    v: npt.NDArray | None = None,
) -> Callable[..., npt.NDArray | DualArray | tuple[npt.NDArray | DualArray, ...]]:
    """Compute the Jacobian of ``func``, or a Jacobian-vector product.

    Wraps ``func`` so that each call evaluates both the function and its
    full Jacobian using forward-mode automatic differentiation with dual
    numbers.  Inputs and outputs may be arrays of any shape.

    Parameters
    ----------
    func : callable
        A function ``f(x, ...) -> y`` where inputs and outputs are scalars
        or arrays of any shape.  When building multi-element outputs inside
        *func*, prefer ``np.stack`` over ``np.array`` (see *Notes*).
    argnums : int or tuple[int, ...], optional
        Positional index (or indices) of the argument(s) to differentiate
        with respect to.  Defaults to ``0``.  When a tuple is given the
        returned function produces a **tuple** of Jacobians, one per index.
    v : ndarray or None, optional
        Direction array for a Jacobian-vector product (JVP).  When
        provided, the returned function computes the directional
        derivative along *v* instead of the full Jacobian.  Must have
        the same shape as the differentiated argument.

    Returns
    -------
    callable
        ``jacobian_func(*args, **kwargs) -> ndarray or tuple[ndarray]``

        * When *argnums* is an ``int``, returns a single Jacobian tensor
          (shape ``y.shape + x.shape``), or the JVP when *v* is given.
        * When *argnums* is a ``tuple``, returns a tuple of Jacobians.

    Raises
    ------
    ValueError
        If *v* and the differentiated argument have mismatched shapes.

    Notes
    -----
    When building multi-element outputs inside ``func``, use
    ``np.stack([...])`` rather than ``np.array([...])``.  Due to a
    limitation in NumPy's dispatch mechanism, ``np.array`` does not
    correctly propagate derivative information through lists of
    intermediate results, which leads to slower fallback behavior and
    a ``UserWarning``.  ``np.stack`` does not have this limitation.

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import jacobian
    >>> f = lambda x: np.stack([x[0]**2, x[0] * x[1]])
    >>> jacobian(f)(np.array([2.0, 3.0]))
    array([[4., 0.],
           [3., 2.]])

    Multi-argument with ``argnums``:

    >>> g = lambda x, y: x**2 * y
    >>> jacobian(g, argnums=0)(np.array(3.0), np.array(2.0))
    np.float64(12.0)
    >>> jacobian(g, argnums=(0, 1))(np.array(3.0), np.array(2.0))
    (np.float64(12.0), np.float64(9.0))
    """
    if isinstance(argnums, int):

        def jacobian_func(*args, **kwargs):
            partial = _argnums_partial(func, args, kwargs, argnums)
            x = _ensure_array(args[argnums])
            if v is not None and not isinstance(x, DualArray):
                if x.shape != v.shape:
                    raise ValueError(
                        f"shape mismatch, x.shape = {x.shape}, v.shape = {v.shape}"
                    )
            return _columnwise_jacobian(partial, x, v)

        return jacobian_func
    else:

        def jacobian_func(*args, **kwargs):
            results = []
            for idx in argnums:
                partial = _argnums_partial(func, args, kwargs, idx)
                x = _ensure_array(args[idx])
                if v is not None and not isinstance(x, DualArray):
                    if x.shape != v.shape:
                        raise ValueError(
                            f"shape mismatch, x.shape = {x.shape}, v.shape = {v.shape}"
                        )
                results.append(_columnwise_jacobian(partial, x, v))
            return tuple(results)

        return jacobian_func


def derivative(
    func: Callable[..., Any], argnums: int | tuple[int, ...] = 0
) -> Callable[..., npt.NDArray | DualArray | tuple[npt.NDArray | DualArray, ...]]:
    """Compute the derivative of a scalar function ``f: R -> R``.

    A thin wrapper around :func:`jacobian` that enforces the scalar-in,
    scalar-out contract and returns a plain scalar rather than a 1x1
    Jacobian.

    Parameters
    ----------
    func : callable
        A function ``f(x, ...) -> y`` where the differentiated argument(s)
        and *y* are scalars.
    argnums : int or tuple[int, ...], optional
        Positional index (or indices) of the argument(s) to differentiate
        with respect to.  Defaults to ``0``.  When a tuple is given the
        returned function produces a **tuple** of partial derivatives.

    Returns
    -------
    callable
        ``df(*args) -> float or tuple[float]`` — evaluates partial
        derivative(s) via forward-mode AD.

    Raises
    ------
    ValueError
        If a differentiated argument is not a scalar (use :func:`gradient`
        for ``R^n -> R``) or if *func* does not return a scalar (use
        :func:`jacobian` for vector-valued functions).

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import derivative
    >>> df = derivative(np.sin)
    >>> df(0.0)  # cos(0) = 1
    1.0

    Multi-argument:

    >>> f = lambda x, y: x**2 * y
    >>> derivative(f, argnums=0)(3.0, 2.0)
    np.float64(12.0)
    >>> derivative(f, argnums=(0, 1))(3.0, 2.0)
    (np.float64(12.0), np.float64(9.0))
    """
    if isinstance(argnums, int):
        _jac = jacobian(func, argnums=argnums)

        def derivative_func(*args, **kwargs):
            if any(isinstance(a, DualArray) for a in args):
                return _jac(*args, **kwargs)
            x = np.asarray(args[argnums])
            if x.ndim != 0:
                raise ValueError(
                    f"derivative expects scalar input (R -> R), "
                    f"got {x.ndim}D array with shape {x.shape}. "
                    f"Use gradient for R^n -> R or jacobian for the "
                    f"general case."
                )
            result = np.squeeze(_jac(*args, **kwargs))
            if result.ndim != 0:
                raise ValueError(
                    f"derivative expects a scalar-valued function (R -> R), "
                    f"but the function returned a result with shape "
                    f"{result.shape}. "
                    f"Use jacobian for vector-valued functions."
                )
            return result

        return derivative_func
    else:
        _jacs = {idx: jacobian(func, argnums=idx) for idx in argnums}

        def derivative_func(*args, **kwargs):
            if any(isinstance(a, DualArray) for a in args):
                return tuple(_jacs[idx](*args, **kwargs) for idx in argnums)
            results = []
            for idx in argnums:
                x = np.asarray(args[idx])
                if x.ndim != 0:
                    raise ValueError(
                        f"derivative expects scalar input (R -> R) for "
                        f"argument {idx}, got {x.ndim}D array with shape "
                        f"{x.shape}. Use gradient for R^n -> R or jacobian "
                        f"for the general case."
                    )
                result = np.squeeze(_jacs[idx](*args, **kwargs))
                if result.ndim != 0:
                    raise ValueError(
                        f"derivative expects a scalar-valued function "
                        f"(R -> R), but the function returned a result with "
                        f"shape {result.shape}. "
                        f"Use jacobian for vector-valued functions."
                    )
                results.append(result)
            return tuple(results)

        return derivative_func


def nth_derivative(
    func: Callable[..., Any], n: int, argnums: int | tuple[int, ...] = 0
) -> Callable[..., Any]:
    """Compute the *n*-th derivative of a scalar function ``f: R -> R``.

    Composes :func:`derivative` *n* times, using nested dual numbers for
    exact higher-order derivatives.

    Parameters
    ----------
    func : callable
        A function ``f(x, ...) -> y`` where the differentiated argument(s)
        and *y* are scalars.
    n : int
        The order of differentiation.  Must be non-negative.
        ``n=0`` returns *func* unchanged, ``n=1`` is equivalent to
        :func:`derivative`.
    argnums : int or tuple[int, ...], optional
        Positional index (or indices) of the argument(s) to differentiate
        with respect to.  Defaults to ``0``.

    Returns
    -------
    callable
        ``f_n(*args) -> float`` — evaluates ``f^(n)`` via forward-mode AD.

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import nth_derivative
    >>> f = lambda x: x ** 4
    >>> nth_derivative(f, 3)(1.0)  # f'''(x) = 24x, at x=1
    24.0

    Cyclic derivatives of sine:

    >>> nth_derivative(np.sin, 4)(0.5)  # sin^(4) = sin
    np.float64(0.479425538604203...)
    >>> np.sin(0.5)
    0.479425538604203...
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    f = func
    for _ in range(n):
        f = derivative(f, argnums=argnums)
    return f


def gradient(
    func: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    *,
    v: npt.NDArray | None = None,
) -> Callable[..., npt.NDArray | DualArray | tuple[npt.NDArray | DualArray, ...]]:
    """Compute the gradient of a scalar-valued function.

    A thin wrapper around :func:`jacobian` that enforces the scalar-output
    contract.  The input *x* may be an array of any shape; the gradient
    will have the same shape.  When a direction array *v* is given,
    returns the directional derivative ``∇f · v`` instead.

    Parameters
    ----------
    func : callable
        A function ``f(x, ...) -> y`` where *x* is an array of any shape
        and *y* is a scalar.
    argnums : int or tuple[int, ...], optional
        Positional index (or indices) of the argument(s) to differentiate
        with respect to.  Defaults to ``0``.  When a tuple is given the
        returned function produces a **tuple** of gradients, one per index.
    v : ndarray or None, optional
        Direction array for a directional derivative.  Must have the same
        shape as the differentiated argument.  When provided, the returned
        function computes ``∇f(x) · v`` (a scalar) instead of the full
        gradient.

    Returns
    -------
    callable
        ``grad_f(*args) -> ndarray or tuple[ndarray]`` — returns
        ``∇f(x)`` (same shape as *x*) when *v* is None, or the
        directional derivative (scalar) when *v* is given.

    Raises
    ------
    ValueError
        If *func* is not scalar-valued.  Use :func:`jacobian` for
        non-scalar-valued functions.

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import gradient
    >>> f = lambda x: x[0]**2 + x[1]**2
    >>> gradient(f)(np.array([3.0, 4.0]))
    array([6., 8.])

    Multi-argument:

    >>> g = lambda x, y: np.sum(x**2) + y**2
    >>> gradient(g, argnums=0)(np.array([3.0, 4.0]), 1.0)
    array([6., 8.])

    Directional derivative along the first axis:

    >>> gradient(f, v=np.array([1.0, 0.0]))(np.array([3.0, 4.0]))
    6.0
    """
    if isinstance(argnums, int):
        _jac = jacobian(func, argnums=argnums, v=v)

        def gradient_func(*args, **kwargs):
            if any(isinstance(a, DualArray) for a in args):
                return _jac(*args, **kwargs)
            x = np.asarray(args[argnums])
            result = _jac(*args, **kwargs)
            result_arr = np.asarray(result)
            if v is None:
                if result_arr.shape != x.shape:
                    raise ValueError(
                        f"gradient expects a scalar-valued function "
                        f"(R^n -> R), but the Jacobian has shape "
                        f"{result_arr.shape} instead of {x.shape}. "
                        f"Use jacobian for vector-valued functions."
                    )
            else:
                if result_arr.ndim != 0:
                    raise ValueError(
                        f"gradient expects a scalar-valued function "
                        f"(R^n -> R), but the directional derivative has "
                        f"shape {result_arr.shape}. "
                        f"Use jacobian for vector-valued functions."
                    )
            return result

        return gradient_func
    else:
        _jacs = {idx: jacobian(func, argnums=idx, v=v) for idx in argnums}

        def gradient_func(*args, **kwargs):
            if any(isinstance(a, DualArray) for a in args):
                return tuple(_jacs[idx](*args, **kwargs) for idx in argnums)
            results = []
            for idx in argnums:
                x = np.asarray(args[idx])
                result = _jacs[idx](*args, **kwargs)
                result_arr = np.asarray(result)
                if v is None:
                    if result_arr.shape != x.shape:
                        raise ValueError(
                            f"gradient expects a scalar-valued function "
                            f"(R^n -> R), but the Jacobian w.r.t. argument "
                            f"{idx} has shape {result_arr.shape} instead of "
                            f"{x.shape}. "
                            f"Use jacobian for vector-valued functions."
                        )
                else:
                    if result_arr.ndim != 0:
                        raise ValueError(
                            f"gradient expects a scalar-valued function "
                            f"(R^n -> R), but the directional derivative "
                            f"w.r.t. argument {idx} has shape "
                            f"{result_arr.shape}. "
                            f"Use jacobian for vector-valued functions."
                        )
                results.append(result)
            return tuple(results)

        return gradient_func


def hessian(
    func: Callable[..., Any], argnums: int | tuple[int, ...] = 0
) -> Callable[..., npt.NDArray | DualArray | tuple[npt.NDArray | DualArray, ...]]:
    """Compute the Hessian of a scalar-valued function.

    Uses nested forward-mode AD by composing :func:`jacobian` with
    :func:`gradient`, giving exact second derivatives without finite
    differences.  For an input of shape *s*, the Hessian is a tensor
    of shape ``s + s``.

    Parameters
    ----------
    func : callable
        A function ``f(x, ...) -> y`` where *x* is an array of any shape
        and *y* is a scalar.
    argnums : int or tuple of exactly 2 ints, optional
        When an ``int``, computes the Hessian ``∂²f/∂xᵢ²`` with respect
        to argument *i*.  When a pair ``(i, j)``, computes the mixed
        Hessian ``∂²f/(∂xᵢ ∂xⱼ)``.  Defaults to ``0``.

    Returns
    -------
    callable
        ``hess_f(*args) -> ndarray`` — for a 1-D input of length *n*,
        returns the ``(n, n)`` Hessian matrix.  For an n-D input of
        shape *s*, returns a tensor of shape ``s + s``.

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import hessian
    >>> f = lambda x: x[0]**2 + 3*x[1]**2
    >>> hessian(f)(np.array([1.0, 1.0]))
    array([[2., 0.],
           [0., 6.]])

    Mixed partial Hessian:

    >>> g = lambda x, y: np.sum(x**2) * y
    >>> hessian(g, argnums=(0, 0))(np.array([1.0, 2.0]), 3.0)
    array([[6., 0.],
           [0., 6.]])
    """
    if isinstance(argnums, int):
        return jacobian(gradient(func, argnums=argnums), argnums=argnums)
    else:
        if len(argnums) != 2:
            raise ValueError(
                f"hessian argnums must be an int or a pair (i, j), "
                f"got tuple of length {len(argnums)}"
            )
        i, j = argnums
        return jacobian(gradient(func, argnums=j), argnums=i)


def hvp(
    func: Callable[..., Any], v: npt.NDArray, argnums: int = 0
) -> Callable[..., npt.NDArray | DualArray | tuple[npt.NDArray | DualArray, ...]]:
    """Compute the Hessian-vector product ``H(x) · v`` without forming ``H``.

    Uses forward-over-forward AD: differentiates the directional derivative
    ``∇f(x) · v`` with respect to *x*, giving ``H(x) · v`` in O(n) time
    rather than the O(n²) cost of forming the full Hessian.

    Parameters
    ----------
    func : callable
        A function ``f(x, ...) -> y`` where *x* is an array of any shape
        and *y* is a scalar.
    v : ndarray
        Direction vector.  Must have the same shape as the differentiated
        argument.
    argnums : int, optional
        Positional index of the argument to differentiate with respect to.
        Defaults to ``0``.

    Returns
    -------
    callable
        ``hvp_func(*args) -> ndarray`` — returns ``H(x) · v`` (same shape
        as the differentiated argument).

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import hvp
    >>> f = lambda x: x[0]**2 + 3*x[1]**2
    >>> hvp(f, np.array([1.0, 0.0]))(np.array([1.0, 1.0]))
    array([2., 0.])

    Equivalent to ``hessian(f)(x) @ v``, but O(n) instead of O(n²):

    >>> from dualpy import hessian
    >>> hessian(f)(np.array([1.0, 1.0])) @ np.array([1.0, 0.0])
    array([2., 0.])
    """
    directional = gradient(func, argnums=argnums, v=v)
    return gradient(directional, argnums=argnums)


def curl(func: Callable[..., Any]) -> Callable[..., npt.NDArray]:
    """Compute the curl of a vector field ``F: R^3 -> R^3``.

    Defined only for three-dimensional vector fields.  Computes the full
    Jacobian via :func:`jacobian` and extracts the curl components::

        curl(F) = [∂F₃/∂x₂ , ∂F₂/∂x₃,
                   ∂F₁/∂x₃ , ∂F₃/∂x₁,
                   ∂F₂/∂x₁ , ∂F₁/∂x₂]

    Parameters
    ----------
    func : callable
        A function ``F(x) -> y`` where both *x* and *y* are 3-element
        arrays.

    Returns
    -------
    callable
        ``curl_F(x) -> ndarray`` — returns the curl vector (shape ``(3,)``).

    Raises
    ------
    ValueError
        If *x* does not have exactly 3 elements.

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import curl
    >>> F = lambda x: np.stack([x[1], -x[0], np.zeros_like(x[0])])
    >>> curl(F)(np.array([1.0, 2.0, 3.0]))
    array([ 0.,  0., -2.])
    """
    _jac = jacobian(func)

    def curl_func(x: npt.ArrayLike) -> npt.NDArray:
        x = _ensure_array(x)
        if x.size != 3:
            raise ValueError("curl is only defined for R^3 -> R^3")
        J = _jac(x)
        return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])

    return curl_func


def divergence(func: Callable[..., Any]) -> Callable[..., np.floating[Any]]:
    """Compute the divergence of a vector field ``F: R^n -> R^n``.

    The divergence is the trace of the Jacobian: ``div(F) = Σᵢ ∂Fᵢ/∂xᵢ``.

    Parameters
    ----------
    func : callable
        A function ``F(x) -> y`` where *x* and *y* are arrays of the same
        length *n*.

    Returns
    -------
    callable
        ``div_F(x) -> float`` — returns the scalar divergence at *x*.

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import divergence
    >>> F = lambda x: x  # identity field
    >>> divergence(F)(np.array([1.0, 2.0, 3.0]))
    3.0
    """
    _jac = jacobian(func)

    def divergence_func(x: npt.ArrayLike) -> np.floating[Any]:
        x = _ensure_array(x)
        J = _jac(x)
        return np.trace(J)

    return divergence_func


def laplacian(func: Callable[..., Any]) -> Callable[..., np.floating[Any]]:
    """Compute the Laplacian of a scalar-valued function.

    The Laplacian is the sum of unmixed second partial derivatives:
    ``Δf = Σᵢ ∂²f/∂xᵢ²``.  Computed via :func:`hessian`, which uses
    nested forward-mode AD for exact second derivatives.  The input may
    be an array of any shape.

    Parameters
    ----------
    func : callable
        A function ``f(x) -> y`` where *x* is an array of any shape and
        *y* is a scalar.

    Returns
    -------
    callable
        ``lap_f(x) -> float`` — returns the scalar Laplacian at *x*.

    Examples
    --------
    >>> import numpy as np
    >>> from dualpy import laplacian
    >>> f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    >>> laplacian(f)(np.array([1.0, 2.0, 3.0]))
    6.0
    """

    def laplacian_func(x: npt.ArrayLike) -> np.floating[Any]:
        H = hessian(func)(x)
        n = np.asarray(x).size
        return np.trace(H.reshape(n, n))

    return laplacian_func
