import numpy as np

from ..core import register_ufunc


@register_ufunc(np.greater.__name__)
def greater(*inputs, **kwargs):
    x, y = inputs
    x_primal = x[0] if isinstance(x, tuple) else x
    y_primal = y[0] if isinstance(y, tuple) else y
    return np.greater(x_primal, y_primal)


@register_ufunc(np.greater_equal.__name__)
def greater_equal(*inputs, **kwargs):
    x, y = inputs
    x_primal = x[0] if isinstance(x, tuple) else x
    y_primal = y[0] if isinstance(y, tuple) else y
    return np.greater_equal(x_primal, y_primal)


@register_ufunc(np.less.__name__)
def less(*inputs, **kwargs):
    x, y = inputs
    x_primal = x[0] if isinstance(x, tuple) else x
    y_primal = y[0] if isinstance(y, tuple) else y
    return np.less(x_primal, y_primal)


@register_ufunc(np.less_equal.__name__)
def less_equal(*inputs, **kwargs):
    x, y = inputs
    x_primal = x[0] if isinstance(x, tuple) else x
    y_primal = y[0] if isinstance(y, tuple) else y
    return np.less_equal(x_primal, y_primal)


@register_ufunc(np.equal.__name__)
def equal(*inputs, **kwargs):
    x, y = inputs
    x_primal = x[0] if isinstance(x, tuple) else x
    y_primal = y[0] if isinstance(y, tuple) else y
    return np.equal(x_primal, y_primal)


@register_ufunc(np.not_equal.__name__)
def not_equal(*inputs, **kwargs):
    x, y = inputs
    x_primal = x[0] if isinstance(x, tuple) else x
    y_primal = y[0] if isinstance(y, tuple) else y
    return np.not_equal(x_primal, y_primal)
