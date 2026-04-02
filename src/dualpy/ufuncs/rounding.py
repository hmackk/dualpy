import numpy as np

from ..core import register_ufunc


@register_ufunc(np.sign.__name__)
def sign(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.sign(x_primal), np.zeros_like(x_primal)


@register_ufunc(np.heaviside.__name__)
def heaviside(*inputs, **kwargs):
    x, h = inputs
    x_primal = x[0] if isinstance(x, tuple) else x
    h_val = h[0] if isinstance(h, tuple) else h
    return np.heaviside(x_primal, h_val), np.zeros_like(
        np.asarray(x_primal, dtype=float)
    )


@register_ufunc(np.floor.__name__)
def floor(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.floor(x_primal), np.zeros_like(x_primal)


@register_ufunc(np.ceil.__name__)
def ceil(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.ceil(x_primal), np.zeros_like(x_primal)


@register_ufunc(np.trunc.__name__)
def trunc(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.trunc(x_primal), np.zeros_like(x_primal)


@register_ufunc(np.rint.__name__)
def rint(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.rint(x_primal), np.zeros_like(x_primal)
