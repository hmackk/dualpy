import numpy as np

from ..core import register_ufunc


@register_ufunc(np.exp.__name__)
def exp(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    exp_primal = np.exp(x_primal)
    exp_tangent = np.exp(x_primal) * x_tangent
    return exp_primal, exp_tangent


@register_ufunc(np.log.__name__)
def log(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    log_primal = np.log(x_primal)
    log_tangent = x_tangent / x_primal
    return log_primal, log_tangent


@register_ufunc(np.sqrt.__name__)
def sqrt(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    sqrt_primal = np.sqrt(x_primal)
    sqrt_tangent = x_tangent / (2 * sqrt_primal)
    return sqrt_primal, sqrt_tangent


@register_ufunc(np.exp2.__name__)
def exp2(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    primal = np.exp2(x_primal)
    return primal, np.log(2) * primal * x_tangent


@register_ufunc(np.expm1.__name__)
def expm1(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.expm1(x_primal), np.exp(x_primal) * x_tangent


@register_ufunc(np.log2.__name__)
def log2(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.log2(x_primal), x_tangent / (x_primal * np.log(2))


@register_ufunc(np.log10.__name__)
def log10(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.log10(x_primal), x_tangent / (x_primal * np.log(10))


@register_ufunc(np.log1p.__name__)
def log1p(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.log1p(x_primal), x_tangent / (1 + x_primal)


@register_ufunc(np.logaddexp.__name__)
def logaddexp(*inputs, **kwargs):
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    primal = np.logaddexp(x_primal, y_primal)
    exp_x = np.exp(x_primal - primal)
    exp_y = np.exp(y_primal - primal)
    tangent = exp_x * x_tangent + exp_y * y_tangent
    return primal, tangent


@register_ufunc(np.logaddexp2.__name__)
def logaddexp2(*inputs, **kwargs):
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    primal = np.logaddexp2(x_primal, y_primal)
    s = np.exp2(x_primal) + np.exp2(y_primal)
    tangent = (np.exp2(x_primal) * x_tangent + np.exp2(y_primal) * y_tangent) / s
    return primal, tangent
