import numpy as np

from ..core import register_ufunc


@register_ufunc(np.add.__name__)
def add(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    return np.add(x[0], y[0]), np.add(x[1], y[1])


@register_ufunc(np.subtract.__name__)
def sub(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    return np.subtract(x[0], y[0]), np.subtract(x[1], y[1])


@register_ufunc(np.multiply.__name__)
def multiply(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    return np.multiply(x_primal, y_primal), np.multiply(
        x_tangent, y_primal
    ) + np.multiply(x_primal, y_tangent)


@register_ufunc(np.divide.__name__)
def divide(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    return np.divide(x[0], y[0]), np.divide(
        (np.multiply(x[1], y[0]) - np.multiply(x[0], y[1])), np.power(y[0], 2)
    )


@register_ufunc(np.power.__name__)
def power(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, n = inputs
    if isinstance(x, tuple):
        x_primal, x_tangent = x
        if isinstance(n, tuple):
            n_primal, n_tangent = n
            primal = np.power(x_primal, n_primal)
            base_term = n_primal * x_tangent / x_primal
            n_tangent_arr = np.asarray(n_tangent)
            if np.any(n_tangent_arr != 0):
                # DualArray ** DualArray: d/dx[f^g] = f^g * (g' ln(f) + g f'/f)
                tangent = primal * (n_tangent * np.log(x_primal) + base_term)
            else:
                tangent = primal * base_term
            return primal, tangent
        # DualArray ** scalar
        return np.power(x_primal, n), n * np.multiply(
            x_tangent, np.power(x_primal, n - 1)
        )
    else:
        # scalar ** DualArray  (rpow path)
        n_primal, n_tangent = n
        primal = np.power(x, n_primal)
        tangent = primal * np.log(x) * n_tangent
        return primal, tangent


@register_ufunc(np.matmul.__name__)
def matmul(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    return x_primal @ y_primal, x_tangent @ y_primal + x_primal @ y_tangent


@register_ufunc(np.negative.__name__)
def negative(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x_primal, x_tangent = inputs[0]
    return np.negative(x_primal), np.negative(x_tangent)


@register_ufunc(np.absolute.__name__)
def absolute(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x_primal, x_tangent = inputs[0]
    return np.absolute(x_primal), np.sign(x_primal) * x_tangent


@register_ufunc(np.square.__name__)
def square(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x_primal, x_tangent = inputs[0]
    return np.square(x_primal), 2 * x_primal * x_tangent


@register_ufunc(np.cbrt.__name__)
def cbrt(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x_primal, x_tangent = inputs[0]
    cbrt_primal = np.cbrt(x_primal)
    return cbrt_primal, x_tangent / (3 * cbrt_primal**2)


@register_ufunc(np.reciprocal.__name__)
def reciprocal(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x_primal, x_tangent = inputs[0]
    return np.reciprocal(x_primal), -x_tangent / x_primal**2


@register_ufunc(np.maximum.__name__)
def maximum(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    mask = x_primal >= y_primal
    primal = np.maximum(x_primal, y_primal)
    tangent = np.where(mask, x_tangent, y_tangent)
    return primal, tangent


@register_ufunc(np.minimum.__name__)
def minimum(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    mask = x_primal <= y_primal
    primal = np.minimum(x_primal, y_primal)
    tangent = np.where(mask, x_tangent, y_tangent)
    return primal, tangent


@register_ufunc(np.positive.__name__)
def positive(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x_primal, x_tangent = inputs[0]
    return np.positive(x_primal), np.positive(x_tangent)


@register_ufunc(np.float_power.__name__)
def float_power(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, n = inputs
    if isinstance(x, tuple):
        x_primal, x_tangent = x
        if isinstance(n, tuple):
            n_primal, n_tangent = n
            primal = np.float_power(x_primal, n_primal)
            base_term = n_primal * x_tangent / x_primal
            n_tangent_arr = np.asarray(n_tangent)
            if np.any(n_tangent_arr != 0):
                tangent = primal * (n_tangent * np.log(x_primal) + base_term)
            else:
                tangent = primal * base_term
            return primal, tangent
        return (
            np.float_power(x_primal, n),
            n * np.multiply(x_tangent, np.float_power(x_primal, n - 1)),
        )
    else:
        n_primal, n_tangent = n
        primal = np.float_power(x, n_primal)
        tangent = primal * np.log(x) * n_tangent
        return primal, tangent


@register_ufunc(np.fabs.__name__)
def fabs(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x_primal, x_tangent = inputs[0]
    return np.fabs(x_primal), np.sign(x_primal) * x_tangent


@register_ufunc(np.remainder.__name__)
def remainder(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    primal = np.remainder(x_primal, y_primal)
    tangent = x_tangent - np.floor(x_primal / y_primal) * y_tangent
    return primal, tangent


@register_ufunc(np.fmod.__name__)
def fmod(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    primal = np.fmod(x_primal, y_primal)
    tangent = x_tangent - np.trunc(x_primal / y_primal) * y_tangent
    return primal, tangent


@register_ufunc(np.copysign.__name__)
def copysign(*inputs, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    x, y = inputs
    x_primal, x_tangent = x
    y_primal = y[0] if isinstance(y, tuple) else y
    primal = np.copysign(x_primal, y_primal)
    tangent = np.sign(y_primal) * np.fabs(x_tangent) * np.sign(x_tangent)
    return primal, tangent
