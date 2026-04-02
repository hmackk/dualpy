import numpy as np

from ..core import DualArray, register_func


@register_func(np.where.__name__)
def where(*args, **kwargs):
    condition = args[0]
    if len(args) < 3:
        return np.where(condition)
    x, y = args[1], args[2]
    if isinstance(condition, DualArray):
        condition = condition.primal
    x_primal = x.primal if isinstance(x, DualArray) else np.asarray(x, dtype=float)
    x_tangent = x.tangent if isinstance(x, DualArray) else np.zeros_like(x_primal)
    y_primal = y.primal if isinstance(y, DualArray) else np.asarray(y, dtype=float)
    y_tangent = y.tangent if isinstance(y, DualArray) else np.zeros_like(y_primal)
    return (
        np.where(condition, x_primal, y_primal),
        np.where(condition, x_tangent, y_tangent),
    )


@register_func(np.clip.__name__)
def clip(*args, **kwargs):
    a = args[0]
    a_min = args[1] if len(args) > 1 else kwargs.get("a_min", None)
    a_max = args[2] if len(args) > 2 else kwargs.get("a_max", None)
    if isinstance(a_min, DualArray):
        a_min = a_min.primal
    if isinstance(a_max, DualArray):
        a_max = a_max.primal
    primal = np.clip(a.primal, a_min, a_max)
    in_bounds = np.ones_like(a.primal, dtype=bool)
    if a_min is not None:
        in_bounds &= a.primal >= a_min
    if a_max is not None:
        in_bounds &= a.primal <= a_max
    tangent = np.where(in_bounds, a.tangent, 0.0)
    return primal, tangent


@register_func(np.sort.__name__)
def sort(*args, **kwargs):
    a = args[0]
    axis = kwargs.get("axis", -1)
    if len(args) > 1:
        axis = args[1]
    indices = np.argsort(a.primal, axis=axis)
    primal = np.take_along_axis(a.primal, indices, axis=axis)
    tangent = np.take_along_axis(a.tangent, indices, axis=axis)
    return primal, tangent


@register_func(np.argsort.__name__)
def argsort(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items()}
    return np.argsort(a.primal, **kw)


@register_func(np.searchsorted.__name__)
def searchsorted(*args, **kwargs):
    a = args[0]
    v = args[1]
    a_primal = a.primal if isinstance(a, DualArray) else a
    v_primal = v.primal if isinstance(v, DualArray) else v
    kw = {k: val for k, val in kwargs.items()}
    return np.searchsorted(a_primal, v_primal, **kw)


@register_func(np.nonzero.__name__)
def nonzero(*args, **kwargs):
    a = args[0]
    return list(np.nonzero(a.primal))


@register_func(np.flatnonzero.__name__)
def flatnonzero(*args, **kwargs):
    a = args[0]
    return np.flatnonzero(a.primal)


@register_func(np.extract.__name__)
def extract(*args, **kwargs):
    condition = args[0]
    a = args[1]
    cond = condition.primal if isinstance(condition, DualArray) else condition
    return np.extract(cond, a.primal), np.extract(cond, a.tangent)
