import numpy as np

from ..core import DualArray, register_func


@register_func(np.diff.__name__)
def diff(*args, **kwargs):
    a = args[0]
    n = args[1] if len(args) > 1 else kwargs.get("n", 1)
    axis = args[2] if len(args) > 2 else kwargs.get("axis", -1)
    kw = {k: v for k, v in kwargs.items() if k not in ("n", "axis")}
    return np.diff(a.primal, n=n, axis=axis, **kw), np.diff(
        a.tangent, n=n, axis=axis, **kw
    )


@register_func(np.convolve.__name__)
def convolve(*args, **kwargs):
    a, b = args[0], args[1]
    a_primal = a.primal if isinstance(a, DualArray) else np.asarray(a, dtype=float)
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a_primal)
    b_primal = b.primal if isinstance(b, DualArray) else np.asarray(b, dtype=float)
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b_primal)
    mode = kwargs.get("mode", "full")
    return (
        np.convolve(a_primal, b_primal, mode=mode),
        np.convolve(a_tangent, b_primal, mode=mode)
        + np.convolve(a_primal, b_tangent, mode=mode),
    )


@register_func(np.correlate.__name__)
def correlate(*args, **kwargs):
    a, b = args[0], args[1]
    a_primal = a.primal if isinstance(a, DualArray) else np.asarray(a, dtype=float)
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a_primal)
    b_primal = b.primal if isinstance(b, DualArray) else np.asarray(b, dtype=float)
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b_primal)
    mode = kwargs.get("mode", "valid")
    return (
        np.correlate(a_primal, b_primal, mode=mode),
        np.correlate(a_tangent, b_primal, mode=mode)
        + np.correlate(a_primal, b_tangent, mode=mode),
    )


@register_func(np.interp.__name__)
def interp(*args, **kwargs):
    x = args[0]
    xp = args[1]
    fp = args[2]
    x_primal = x.primal if isinstance(x, DualArray) else np.asarray(x, dtype=float)
    x_tangent = x.tangent if isinstance(x, DualArray) else np.zeros_like(x_primal)
    xp_val = xp.primal if isinstance(xp, DualArray) else np.asarray(xp, dtype=float)
    fp_primal = fp.primal if isinstance(fp, DualArray) else np.asarray(fp, dtype=float)
    fp_tangent = fp.tangent if isinstance(fp, DualArray) else np.zeros_like(fp_primal)

    primal = np.interp(x_primal, xp_val, fp_primal)

    slopes = np.zeros_like(fp_primal)
    slopes[:-1] = np.diff(fp_primal) / np.diff(xp_val)
    indices = np.searchsorted(xp_val, x_primal, side="right") - 1
    indices = np.clip(indices, 0, len(xp_val) - 2)
    dfdx = slopes[indices]
    t_from_x = dfdx * x_tangent

    t_from_fp = np.interp(x_primal, xp_val, fp_tangent)

    return primal, t_from_x + t_from_fp


@register_func(np.trapezoid.__name__)
def trapezoid(*args, **kwargs):
    y = args[0]
    y_primal = y.primal if isinstance(y, DualArray) else np.asarray(y, dtype=float)
    y_tangent = y.tangent if isinstance(y, DualArray) else np.zeros_like(y_primal)

    x = args[1] if len(args) > 1 else kwargs.get("x", None)
    dx = kwargs.get("dx", 1.0)
    axis = kwargs.get("axis", -1)

    if x is not None:
        x_primal = x.primal if isinstance(x, DualArray) else np.asarray(x, dtype=float)
        x_tangent = x.tangent if isinstance(x, DualArray) else np.zeros_like(x_primal)
        p = np.trapezoid(y_primal, x=x_primal, axis=axis)
        t_from_y = np.trapezoid(y_tangent, x=x_primal, axis=axis)
        ax = axis % y_primal.ndim
        slc_lo = [slice(None)] * y_primal.ndim
        slc_hi = [slice(None)] * y_primal.ndim
        slc_lo[ax] = slice(None, -1)
        slc_hi[ax] = slice(1, None)
        y_avg = (y_primal[tuple(slc_lo)] + y_primal[tuple(slc_hi)]) / 2.0
        dx_tangent = np.diff(x_tangent, axis=ax)
        t_from_x = np.sum(y_avg * dx_tangent, axis=ax)
        t = t_from_y + t_from_x
    else:
        p = np.trapezoid(y_primal, dx=dx, axis=axis)
        t = np.trapezoid(y_tangent, dx=dx, axis=axis)
    return p, t


@register_func(np.sinc.__name__)
def sinc(*args, **kwargs):
    a = args[0]
    x_primal = a.primal if isinstance(a, DualArray) else np.asarray(a, dtype=float)
    x_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(x_primal)
    primal = np.sinc(x_primal)
    pix = np.pi * x_primal
    near_zero = np.abs(x_primal) < 1e-15
    safe_x = np.where(near_zero, np.ones_like(x_primal), x_primal)
    safe_pix = np.where(near_zero, np.ones_like(pix), pix)
    deriv = np.where(
        near_zero,
        np.zeros_like(x_primal),
        (np.cos(safe_pix) / safe_x) - (np.sin(safe_pix) / (np.pi * safe_x**2)),
    )
    tangent = deriv * x_tangent
    return primal, tangent


@register_func(np.gradient.__name__)
def numpy_gradient(*args, **kwargs):
    a = args[0]
    rest_primals = []
    for r in args[1:]:
        if isinstance(r, DualArray):
            rest_primals.append(r.primal)
        else:
            rest_primals.append(r)
    p_results = np.gradient(a.primal, *rest_primals, **kwargs)
    t_results = np.gradient(a.tangent, *rest_primals, **kwargs)
    if isinstance(p_results, list | tuple) and not isinstance(p_results, np.ndarray):
        return [DualArray(p, t) for p, t in zip(p_results, t_results, strict=True)]
    return DualArray(p_results, t_results)
