import numpy as np

from ..core import DualArray, register_func


@register_func(np.array.__name__)
def array(*args, **kwargs):
    x = args[0]
    primals = [obj.primal for obj in x]
    tangents = [obj.tangent for obj in x]
    return np.array(primals), np.array(tangents)


@register_func(np.zeros_like.__name__)
def zeros_like(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "subok"}
    return np.zeros_like(a.primal, **kw), np.zeros_like(a.primal, **kw)


@register_func(np.ones_like.__name__)
def ones_like(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "subok"}
    return np.ones_like(a.primal, **kw), np.zeros_like(a.primal, **kw)


@register_func(np.full_like.__name__)
def full_like(*args, **kwargs):
    a = args[0]
    fill_value = args[1] if len(args) > 1 else kwargs["fill_value"]
    kw = {k: v for k, v in kwargs.items() if k not in ("fill_value", "subok")}
    return (
        np.full_like(a.primal, fill_value, **kw),
        np.zeros_like(a.primal, **kw),
    )


@register_func(np.copy.__name__)
def copy(*args, **kwargs):
    a = args[0]
    return np.copy(a.primal), np.copy(a.tangent)


@register_func(np.linspace.__name__)
def linspace(*args, **kwargs):
    start = args[0]
    stop = args[1]
    s_primal = start.primal if isinstance(start, DualArray) else start
    e_primal = stop.primal if isinstance(stop, DualArray) else stop
    rest_args = args[2:]
    result = np.linspace(s_primal, e_primal, *rest_args, **kwargs)
    return result, np.zeros_like(result)


@register_func(np.arange.__name__)
def arange(*args, **kwargs):
    raw_args = []
    for a in args:
        raw_args.append(a.primal if isinstance(a, DualArray) else a)
    result = np.arange(*raw_args, **kwargs)
    return result, np.zeros_like(result)


@register_func(np.eye.__name__)
def eye(*args, **kwargs):
    result = np.eye(*args, **kwargs)
    return result, np.zeros_like(result)


@register_func(np.identity.__name__)
def identity(*args, **kwargs):
    result = np.identity(*args, **kwargs)
    return result, np.zeros_like(result)


@register_func(np.empty_like.__name__)
def empty_like(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "subok"}
    return np.zeros_like(a.primal, **kw), np.zeros_like(a.primal, **kw)


@register_func(np.diag.__name__)
def diag(*args, **kwargs):
    a = args[0]
    k = args[1] if len(args) > 1 else kwargs.get("k", 0)
    return np.diag(a.primal, k), np.diag(a.tangent, k)


@register_func(np.diagonal.__name__)
def diagonal(*args, **kwargs):
    a = args[0]
    kw = {k_: v for k_, v in kwargs.items() if k_ != "out"}
    offset = args[1] if len(args) > 1 else kw.pop("offset", 0)
    axis1 = args[2] if len(args) > 2 else kw.pop("axis1", 0)
    axis2 = args[3] if len(args) > 3 else kw.pop("axis2", 1)
    return (
        np.diagonal(a.primal, offset, axis1, axis2),
        np.diagonal(a.tangent, offset, axis1, axis2),
    )


@register_func(np.meshgrid.__name__)
def meshgrid(*args, **kwargs):
    primals = []
    for a in args:
        if isinstance(a, DualArray):
            primals.append(a.primal)
        else:
            primals.append(np.asarray(a))
    grids = np.meshgrid(*primals, **kwargs)
    result = []
    for g in grids:
        result.append(DualArray(g, np.zeros_like(g)))
    return result


@register_func(np.triu.__name__)
def triu(*args, **kwargs):
    a = args[0]
    k = args[1] if len(args) > 1 else kwargs.get("k", 0)
    return np.triu(a.primal, k), np.triu(a.tangent, k)


@register_func(np.tril.__name__)
def tril(*args, **kwargs):
    a = args[0]
    k = args[1] if len(args) > 1 else kwargs.get("k", 0)
    return np.tril(a.primal, k), np.tril(a.tangent, k)
