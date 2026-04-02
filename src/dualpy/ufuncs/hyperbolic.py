import numpy as np

from ..core import register_ufunc


@register_ufunc(np.sinh.__name__)
def sinh(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.sinh(x_primal), np.cosh(x_primal) * x_tangent


@register_ufunc(np.cosh.__name__)
def cosh(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.cosh(x_primal), np.sinh(x_primal) * x_tangent


@register_ufunc(np.tanh.__name__)
def tanh(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    tanh_primal = np.tanh(x_primal)
    return tanh_primal, (1 - tanh_primal**2) * x_tangent


@register_ufunc(np.arcsinh.__name__)
def arcsinh(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.arcsinh(x_primal), x_tangent / np.sqrt(x_primal**2 + 1)


@register_ufunc(np.arccosh.__name__)
def arccosh(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.arccosh(x_primal), x_tangent / np.sqrt(x_primal**2 - 1)


@register_ufunc(np.arctanh.__name__)
def arctanh(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.arctanh(x_primal), x_tangent / (1 - x_primal**2)
