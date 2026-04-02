import numpy as np

from ..core import register_ufunc


def _primal(x):
    return x[0] if isinstance(x, tuple) else x


@register_ufunc(np.logical_and.__name__)
def logical_and(*inputs, **kwargs):
    x, y = inputs
    return np.logical_and(_primal(x), _primal(y))


@register_ufunc(np.logical_or.__name__)
def logical_or(*inputs, **kwargs):
    x, y = inputs
    return np.logical_or(_primal(x), _primal(y))


@register_ufunc(np.logical_xor.__name__)
def logical_xor(*inputs, **kwargs):
    x, y = inputs
    return np.logical_xor(_primal(x), _primal(y))


@register_ufunc(np.logical_not.__name__)
def logical_not(*inputs, **kwargs):
    return np.logical_not(_primal(inputs[0]))


@register_ufunc(np.isnan.__name__)
def isnan(*inputs, **kwargs):
    return np.isnan(_primal(inputs[0]))


@register_ufunc(np.isinf.__name__)
def isinf(*inputs, **kwargs):
    return np.isinf(_primal(inputs[0]))


@register_ufunc(np.isfinite.__name__)
def isfinite(*inputs, **kwargs):
    return np.isfinite(_primal(inputs[0]))


@register_ufunc(np.signbit.__name__)
def signbit(*inputs, **kwargs):
    return np.signbit(_primal(inputs[0]))
