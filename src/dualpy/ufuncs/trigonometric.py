import numpy as np

from ..core import register_ufunc


@register_ufunc(np.sin.__name__)
def sin(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    sin_primal = np.sin(x_primal)
    sin_tangent = np.cos(x_primal) * x_tangent
    return sin_primal, sin_tangent


@register_ufunc(np.cos.__name__)
def cos(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    cos_primal = np.cos(x_primal)
    cos_tangent = -np.sin(x_primal) * x_tangent
    return cos_primal, cos_tangent


@register_ufunc(np.tan.__name__)
def tan(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    tan_primal = np.tan(x_primal)
    tan_tangent = (1 + tan_primal**2) * x_tangent
    return tan_primal, tan_tangent


@register_ufunc(np.arcsin.__name__)
def arcsin(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    arcsin_primal = np.arcsin(x_primal)
    arcsin_tangent = x_tangent / np.sqrt(1 - x_primal**2)
    return arcsin_primal, arcsin_tangent


@register_ufunc(np.arccos.__name__)
def arccos(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    arccos_primal = np.arccos(x_primal)
    arccos_tangent = -x_tangent / np.sqrt(1 - x_primal**2)
    return arccos_primal, arccos_tangent


@register_ufunc(np.arctan.__name__)
def arctan(*inputs, **kwargs):
    x = inputs[0]
    x_primal, x_tangent = x
    arctan_primal = np.arctan(x_primal)
    arctan_tangent = x_tangent / (1 + x_primal**2)
    return arctan_primal, arctan_tangent


@register_ufunc(np.arctan2.__name__)
def arctan2(*inputs, **kwargs):
    y, x = inputs
    y_primal, y_tangent = y
    x_primal, x_tangent = x
    denom = x_primal**2 + y_primal**2
    primal = np.arctan2(y_primal, x_primal)
    tangent = (x_primal * y_tangent - y_primal * x_tangent) / denom
    return primal, tangent


@register_ufunc(np.hypot.__name__)
def hypot(*inputs, **kwargs):
    x, y = inputs
    x_primal, x_tangent = x
    y_primal, y_tangent = y
    primal = np.hypot(x_primal, y_primal)
    safe = np.where(primal == 0, np.ones_like(primal), primal)
    tangent = np.where(
        primal == 0,
        np.zeros_like(primal),
        (x_primal * x_tangent + y_primal * y_tangent) / safe,
    )
    return primal, tangent


@register_ufunc(np.degrees.__name__)
def degrees(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.degrees(x_primal), x_tangent * (180.0 / np.pi)


@register_ufunc(np.rad2deg.__name__)
def rad2deg(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.rad2deg(x_primal), x_tangent * (180.0 / np.pi)


@register_ufunc(np.radians.__name__)
def radians(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.radians(x_primal), x_tangent * (np.pi / 180.0)


@register_ufunc(np.deg2rad.__name__)
def deg2rad(*inputs, **kwargs):
    x_primal, x_tangent = inputs[0]
    return np.deg2rad(x_primal), x_tangent * (np.pi / 180.0)
