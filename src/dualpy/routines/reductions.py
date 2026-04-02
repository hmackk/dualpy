import numpy as np

from ..core import DualArray, register_func


@register_func(np.sum.__name__)
def sum_(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.sum(a.primal, **kw), np.sum(a.tangent, **kw)


@register_func(np.mean.__name__)
def mean(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.mean(a.primal, **kw), np.mean(a.tangent, **kw)


@register_func(np.prod.__name__)
def prod(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    p = np.prod(a.primal, **kw)
    if axis is None:
        x = a.primal.ravel()
        dx = a.tangent.ravel()
    else:
        x = np.moveaxis(a.primal, axis, -1)
        dx = np.moveaxis(a.tangent, axis, -1)
    n = x.shape[-1]
    left = np.ones_like(x)
    right = np.ones_like(x)
    for i in range(1, n):
        left[..., i] = left[..., i - 1] * x[..., i - 1]
    for i in range(n - 2, -1, -1):
        right[..., i] = right[..., i + 1] * x[..., i + 1]
    t = np.sum(left * right * dx, axis=-1)
    if axis is not None and keepdims:
        t = np.expand_dims(t, axis=axis)
    return p, t


@register_func(np.max.__name__)
def max_(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k not in ("out", "initial", "where")}
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    p = np.max(a.primal, **kw)
    if axis is None:
        idx = np.argmax(a.primal)
        t = a.tangent.flat[idx]
    else:
        idx = np.expand_dims(np.argmax(a.primal, axis=axis), axis=axis)
        t = np.take_along_axis(a.tangent, idx, axis=axis)
        if not keepdims:
            t = np.squeeze(t, axis=axis)
    return p, t


@register_func(np.min.__name__)
def min_(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k not in ("out", "initial", "where")}
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    p = np.min(a.primal, **kw)
    if axis is None:
        idx = np.argmin(a.primal)
        t = a.tangent.flat[idx]
    else:
        idx = np.expand_dims(np.argmin(a.primal, axis=axis), axis=axis)
        t = np.take_along_axis(a.tangent, idx, axis=axis)
        if not keepdims:
            t = np.squeeze(t, axis=axis)
    return p, t


@register_func(np.var.__name__)
def var(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    p = np.var(a.primal, **kw)
    ddof = kw.get("ddof", 0)
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    diff = a.primal - np.mean(a.primal, axis=axis, keepdims=True)
    n = a.primal.size if axis is None else a.primal.shape[axis]
    t = 2 * np.sum(diff * a.tangent, axis=axis, keepdims=keepdims) / (n - ddof)
    return p, t


@register_func(np.std.__name__)
def std(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    ddof = kw.get("ddof", 0)
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    s = np.std(a.primal, **kw)
    diff = a.primal - np.mean(a.primal, axis=axis, keepdims=True)
    n = a.primal.size if axis is None else a.primal.shape[axis]
    dvar = 2 * np.sum(diff * a.tangent, axis=axis, keepdims=keepdims) / (n - ddof)
    safe_s = np.where(s == 0, np.ones_like(s), s)
    t = np.where(s == 0, 0.0, dvar / (2 * safe_s))
    return s, t


@register_func(np.cumsum.__name__)
def cumsum(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.cumsum(a.primal, **kw), np.cumsum(a.tangent, **kw)


@register_func(np.cumprod.__name__)
def cumprod(*args, **kwargs):
    a = args[0]
    axis = kwargs.get("axis", None)
    p = np.cumprod(a.primal, axis=axis)
    x = a.primal
    dx = a.tangent
    if axis is None:
        x_flat = x.ravel()
        dx_flat = dx.ravel()
        n = x_flat.size
        t = np.empty(n, dtype=p.dtype)
        t[0] = dx_flat[0]
        for i in range(1, n):
            t[i] = t[i - 1] * x_flat[i] + p[i - 1] * dx_flat[i]
        return p, t
    nd = x.ndim
    ax = axis % nd
    n = x.shape[ax]
    t = np.empty_like(p)
    idx_prev = [slice(None)] * nd
    idx_cur = [slice(None)] * nd
    idx_cur[ax] = 0
    t[tuple(idx_cur)] = dx[tuple(idx_cur)]
    for i in range(1, n):
        idx_prev[ax] = i - 1
        idx_cur[ax] = i
        t[tuple(idx_cur)] = (
            t[tuple(idx_prev)] * x[tuple(idx_cur)]
            + p[tuple(idx_prev)] * dx[tuple(idx_cur)]
        )
    return p, t


@register_func(np.argmax.__name__)
def argmax(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.argmax(a.primal, **kw)


@register_func(np.argmin.__name__)
def argmin(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.argmin(a.primal, **kw)


def _nan_mask(primal):
    return ~np.isnan(primal)


@register_func(np.nansum.__name__)
def nansum(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    tangent_safe = np.where(_nan_mask(a.primal), a.tangent, 0.0)
    return np.nansum(a.primal, **kw), np.sum(tangent_safe, **kw)


@register_func(np.nanmean.__name__)
def nanmean(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    mask = _nan_mask(a.primal)
    tangent_safe = np.where(mask, a.tangent, 0.0)
    count = np.sum(mask, axis=axis, keepdims=keepdims).astype(float)
    count = np.where(count == 0, 1.0, count)
    return np.nanmean(a.primal, **kw), np.sum(
        tangent_safe, axis=axis, keepdims=keepdims
    ) / count


@register_func(np.nanvar.__name__)
def nanvar(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    ddof = kw.get("ddof", 0)
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    p = np.nanvar(a.primal, **kw)
    mask = _nan_mask(a.primal)
    mean_val = np.nanmean(a.primal, axis=axis, keepdims=True)
    diff = np.where(mask, a.primal - mean_val, 0.0)
    tangent_safe = np.where(mask, a.tangent, 0.0)
    n = np.sum(mask, axis=axis, keepdims=keepdims).astype(float)
    n = np.where(n == 0, 1.0, n)
    t = 2 * np.sum(diff * tangent_safe, axis=axis, keepdims=keepdims) / (n - ddof)
    return p, t


@register_func(np.nanstd.__name__)
def nanstd(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    ddof = kw.get("ddof", 0)
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    s = np.nanstd(a.primal, **kw)
    mask = _nan_mask(a.primal)
    mean_val = np.nanmean(a.primal, axis=axis, keepdims=True)
    diff = np.where(mask, a.primal - mean_val, 0.0)
    tangent_safe = np.where(mask, a.tangent, 0.0)
    n = np.sum(mask, axis=axis, keepdims=keepdims).astype(float)
    n = np.where(n == 0, 1.0, n)
    dvar = 2 * np.sum(diff * tangent_safe, axis=axis, keepdims=keepdims) / (n - ddof)
    safe_s = np.where(s == 0, np.ones_like(s), s)
    t = np.where(s == 0, 0.0, dvar / (2 * safe_s))
    return s, t


@register_func(np.nanmax.__name__)
def nanmax(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k not in ("out", "initial", "where")}
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    filled = np.where(_nan_mask(a.primal), a.primal, -np.inf)
    p = np.nanmax(a.primal, **kw)
    if axis is None:
        idx = np.argmax(filled)
        t = a.tangent.flat[idx]
    else:
        idx = np.expand_dims(np.argmax(filled, axis=axis), axis=axis)
        t = np.take_along_axis(a.tangent, idx, axis=axis)
        if not keepdims:
            t = np.squeeze(t, axis=axis)
    return p, t


@register_func(np.nanmin.__name__)
def nanmin(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k not in ("out", "initial", "where")}
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    filled = np.where(_nan_mask(a.primal), a.primal, np.inf)
    p = np.nanmin(a.primal, **kw)
    if axis is None:
        idx = np.argmin(filled)
        t = a.tangent.flat[idx]
    else:
        idx = np.expand_dims(np.argmin(filled, axis=axis), axis=axis)
        t = np.take_along_axis(a.tangent, idx, axis=axis)
        if not keepdims:
            t = np.squeeze(t, axis=axis)
    return p, t


@register_func(np.average.__name__)
def average(*args, **kwargs):
    a = args[0]
    weights = kwargs.get("weights", None)
    axis = kwargs.get("axis", None)
    returned = kwargs.get("returned", False)
    keepdims = kwargs.get("keepdims", False)

    a_primal = a.primal
    a_tangent = a.tangent

    if weights is None:
        p = np.average(a_primal, axis=axis)
        t = np.mean(a_tangent, axis=axis)
    else:
        if isinstance(weights, DualArray):
            w_primal = weights.primal
            w_tangent = weights.tangent
        else:
            w_primal = np.asarray(weights, dtype=float)
            w_tangent = np.zeros_like(w_primal)
        wsum = np.sum(w_primal, axis=axis)
        p = np.sum(a_primal * w_primal, axis=axis) / wsum
        t = (
            np.sum(a_tangent * w_primal + a_primal * w_tangent, axis=axis) * wsum
            - np.sum(a_primal * w_primal, axis=axis) * np.sum(w_tangent, axis=axis)
        ) / wsum**2

    if keepdims and axis is not None:
        p = np.expand_dims(p, axis=axis)
        t = np.expand_dims(t, axis=axis)

    if returned:
        if weights is None:
            sw = np.asarray(
                a_primal.size if axis is None else a_primal.shape[axis], dtype=float
            )
        else:
            sw = np.sum(w_primal, axis=axis)
            if keepdims and axis is not None:
                sw = np.expand_dims(sw, axis=axis)
        return [DualArray(p, t), sw]
    return p, t


@register_func(np.median.__name__)
def median(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    axis = kw.get("axis", None)
    keepdims = kw.get("keepdims", False)
    p = np.median(a.primal, **kw)
    if axis is None:
        flat = a.primal.ravel()
        flat_t = a.tangent.ravel()
        sorted_idx = np.argsort(flat)
        n = flat.size
        if n % 2 == 1:
            t = flat_t[sorted_idx[n // 2]]
        else:
            t = (flat_t[sorted_idx[n // 2 - 1]] + flat_t[sorted_idx[n // 2]]) / 2.0
        t = np.asarray(t)
    else:
        sorted_idx = np.argsort(a.primal, axis=axis)
        n = a.primal.shape[axis]
        if n % 2 == 1:
            mid = np.take_along_axis(
                sorted_idx,
                np.expand_dims(
                    np.full(
                        [s for i, s in enumerate(a.primal.shape) if i != axis], n // 2
                    ),
                    axis=axis,
                ),
                axis=axis,
            )
            t = np.take_along_axis(a.tangent, mid, axis=axis)
        else:
            mid_lo = np.take_along_axis(
                sorted_idx,
                np.expand_dims(
                    np.full(
                        [s for i, s in enumerate(a.primal.shape) if i != axis],
                        n // 2 - 1,
                    ),
                    axis=axis,
                ),
                axis=axis,
            )
            mid_hi = np.take_along_axis(
                sorted_idx,
                np.expand_dims(
                    np.full(
                        [s for i, s in enumerate(a.primal.shape) if i != axis], n // 2
                    ),
                    axis=axis,
                ),
                axis=axis,
            )
            t = (
                np.take_along_axis(a.tangent, mid_lo, axis=axis)
                + np.take_along_axis(a.tangent, mid_hi, axis=axis)
            ) / 2.0
        if not keepdims:
            t = np.squeeze(t, axis=axis)
    return p, t


@register_func(np.all.__name__)
def all_(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.all(a.primal, **kw)


@register_func(np.any.__name__)
def any_(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.any(a.primal, **kw)


@register_func(np.count_nonzero.__name__)
def count_nonzero(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items()}
    return np.count_nonzero(a.primal, **kw)
