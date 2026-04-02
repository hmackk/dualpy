import numpy as np

from ..core import DualArray, register_func


@register_func(np.dot.__name__)
def dot(*args, **kwargs):
    a, b = args[0], args[1]
    a_primal = a.primal if isinstance(a, DualArray) else a
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a)
    b_primal = b.primal if isinstance(b, DualArray) else b
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b)
    return (
        np.dot(a_primal, b_primal),
        np.dot(a_tangent, b_primal) + np.dot(a_primal, b_tangent),
    )


@register_func(np.inner.__name__)
def inner(*args, **kwargs):
    a, b = args[0], args[1]
    a_primal = a.primal if isinstance(a, DualArray) else a
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a)
    b_primal = b.primal if isinstance(b, DualArray) else b
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b)
    return (
        np.inner(a_primal, b_primal),
        np.inner(a_tangent, b_primal) + np.inner(a_primal, b_tangent),
    )


@register_func(np.outer.__name__)
def outer(*args, **kwargs):
    a, b = args[0], args[1]
    a_primal = a.primal if isinstance(a, DualArray) else a
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a)
    b_primal = b.primal if isinstance(b, DualArray) else b
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b)
    return (
        np.outer(a_primal, b_primal),
        np.outer(a_tangent, b_primal) + np.outer(a_primal, b_tangent),
    )


@register_func(np.tensordot.__name__)
def tensordot(*args, **kwargs):
    a, b = args[0], args[1]
    axes = args[2] if len(args) > 2 else kwargs.get("axes", 2)
    a_primal = a.primal if isinstance(a, DualArray) else a
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a)
    b_primal = b.primal if isinstance(b, DualArray) else b
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b)
    return (
        np.tensordot(a_primal, b_primal, axes=axes),
        np.tensordot(a_tangent, b_primal, axes=axes)
        + np.tensordot(a_primal, b_tangent, axes=axes),
    )


@register_func(np.einsum.__name__)
def einsum(*args, **kwargs):
    subscripts = args[0]
    operands = list(args[1:])
    primals = []
    tangents = []
    for op in operands:
        if isinstance(op, DualArray):
            primals.append(op.primal)
            tangents.append(op.tangent)
        else:
            primals.append(op)
            tangents.append(np.zeros_like(op))
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    primal = np.einsum(subscripts, *primals, **kw)
    tangent = np.zeros_like(primal)
    for i in range(len(primals)):
        parts = list(primals)
        parts[i] = tangents[i]
        tangent = tangent + np.einsum(subscripts, *parts, **kw)
    return primal, tangent


@register_func(np.trace.__name__)
def trace(*args, **kwargs):
    a = args[0]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.trace(a.primal, **kw), np.trace(a.tangent, **kw)


@register_func(np.cross.__name__)
def cross(*args, **kwargs):
    a, b = args[0], args[1]
    a_primal = a.primal if isinstance(a, DualArray) else a
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a)
    b_primal = b.primal if isinstance(b, DualArray) else b
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b)
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return (
        np.cross(a_primal, b_primal, **kw),
        np.cross(a_tangent, b_primal, **kw) + np.cross(a_primal, b_tangent, **kw),
    )


@register_func(np.vdot.__name__)
def vdot(*args, **kwargs):
    a, b = args[0], args[1]
    a_primal = a.primal if isinstance(a, DualArray) else np.asarray(a, dtype=float)
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a_primal)
    b_primal = b.primal if isinstance(b, DualArray) else np.asarray(b, dtype=float)
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b_primal)
    return (
        np.vdot(a_primal, b_primal),
        np.vdot(a_tangent, b_primal) + np.vdot(a_primal, b_tangent),
    )


@register_func(np.kron.__name__)
def kron(*args, **kwargs):
    a, b = args[0], args[1]
    a_primal = a.primal if isinstance(a, DualArray) else np.asarray(a, dtype=float)
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a_primal)
    b_primal = b.primal if isinstance(b, DualArray) else np.asarray(b, dtype=float)
    b_tangent = b.tangent if isinstance(b, DualArray) else np.zeros_like(b_primal)
    return (
        np.kron(a_primal, b_primal),
        np.kron(a_tangent, b_primal) + np.kron(a_primal, b_tangent),
    )


@register_func(np.linalg.norm.__name__)
def norm(*args, **kwargs):
    a = args[0]
    a_primal = a.primal if isinstance(a, DualArray) else np.asarray(a, dtype=float)
    a_tangent = a.tangent if isinstance(a, DualArray) else np.zeros_like(a_primal)
    ord_ = kwargs.get("ord", None)
    if len(args) > 1:
        ord_ = args[1]
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    p = np.linalg.norm(a_primal, ord=ord_, axis=axis, keepdims=keepdims)

    if axis is None and a_primal.ndim <= 1:
        if ord_ is None or ord_ == 2:
            safe_p = np.where(p == 0, np.ones_like(p), p)
            t = np.where(p == 0, 0.0, np.sum(a_primal * a_tangent) / safe_p)
        elif ord_ == 1:
            t = np.sum(np.sign(a_primal) * a_tangent)
        elif ord_ == np.inf:
            idx = np.argmax(np.abs(a_primal))
            t = np.sign(a_primal.flat[idx]) * a_tangent.flat[idx]
        elif ord_ == -np.inf:
            idx = np.argmin(np.abs(a_primal))
            t = np.sign(a_primal.flat[idx]) * a_tangent.flat[idx]
        else:
            safe_p = np.where(p == 0, np.ones_like(p), p)
            t = np.where(
                p == 0,
                0.0,
                np.sum(np.sign(a_primal) * np.abs(a_primal) ** (ord_ - 1) * a_tangent)
                / safe_p ** (ord_ - 1),
            )
    elif axis is not None:
        if ord_ is None or ord_ == 2:
            safe_p = np.where(p == 0, np.ones_like(p), p)
            t_val = np.sum(a_primal * a_tangent, axis=axis, keepdims=keepdims) / (
                safe_p if keepdims else safe_p
            )
            t = np.where(p == 0 if keepdims else p == 0, 0.0, t_val)
        elif ord_ == 1:
            t = np.sum(np.sign(a_primal) * a_tangent, axis=axis, keepdims=keepdims)
        else:
            safe_p = np.where(p == 0, np.ones_like(p), p)
            t = np.sum(
                np.sign(a_primal) * np.abs(a_primal) ** (ord_ - 1) * a_tangent,
                axis=axis,
                keepdims=keepdims,
            ) / safe_p ** (ord_ - 1)
    else:
        safe_p = np.where(p == 0, np.ones_like(p), p)
        t = np.where(p == 0, 0.0, np.sum(a_primal * a_tangent) / safe_p)
    return p, t


@register_func(np.linalg.det.__name__)
def det(*args, **kwargs):
    a = args[0]
    p = np.linalg.det(a.primal)
    inv_a = np.linalg.inv(a.primal)
    if a.primal.ndim == 2:
        t = p * np.trace(inv_a @ a.tangent)
    else:
        t = p * np.trace(inv_a @ a.tangent, axis1=-2, axis2=-1)
    return p, t


@register_func(np.linalg.inv.__name__)
def inv(*args, **kwargs):
    a = args[0]
    inv_primal = np.linalg.inv(a.primal)
    t = -inv_primal @ a.tangent @ inv_primal
    return inv_primal, t


@register_func(np.linalg.solve.__name__)
def solve(*args, **kwargs):
    a_arg, b_arg = args[0], args[1]
    a_primal = (
        a_arg.primal if isinstance(a_arg, DualArray) else np.asarray(a_arg, dtype=float)
    )
    a_tangent = (
        a_arg.tangent if isinstance(a_arg, DualArray) else np.zeros_like(a_primal)
    )
    b_primal = (
        b_arg.primal if isinstance(b_arg, DualArray) else np.asarray(b_arg, dtype=float)
    )
    b_tangent = (
        b_arg.tangent if isinstance(b_arg, DualArray) else np.zeros_like(b_primal)
    )
    x = np.linalg.solve(a_primal, b_primal)
    dx = np.linalg.solve(a_primal, b_tangent - a_tangent @ x)
    return x, dx


@register_func(np.linalg.cholesky.__name__)
def cholesky(*args, **kwargs):
    a = args[0]
    L = np.linalg.cholesky(a.primal)
    dA = a.tangent
    n = L.shape[-1]
    L_inv = np.linalg.inv(L)
    S = L_inv @ dA @ L_inv.T
    S_lower = np.tril(S)
    S_lower_diag_half = S_lower.copy()
    idx = np.arange(n)
    S_lower_diag_half[..., idx, idx] = S_lower[..., idx, idx] / 2.0
    dL = L @ S_lower_diag_half
    return L, dL


@register_func(np.linalg.eigh.__name__)
def eigh(*args, **kwargs):
    a = args[0]
    UPLO = kwargs.get("UPLO", "L")
    w, V = np.linalg.eigh(a.primal, UPLO=UPLO)
    dA = a.tangent
    dw = np.sum(V * (dA @ V), axis=-2)
    n = w.shape[-1]
    E = w[..., np.newaxis, :] - w[..., :, np.newaxis]
    mask = np.eye(n, dtype=bool)
    E_safe = np.where(mask, np.ones_like(E), E)
    F = np.where(mask, np.zeros_like(E), 1.0 / E_safe)
    VtdAV = np.swapaxes(V, -2, -1) @ dA @ V
    dV = V @ (F * VtdAV)
    return [DualArray(w, dw), DualArray(V, dV)]


@register_func(np.linalg.svd.__name__)
def svd(*args, **kwargs):
    a = args[0]
    full_matrices = kwargs.get("full_matrices", True)
    compute_uv = kwargs.get("compute_uv", True)
    U, s, Vt = np.linalg.svd(a.primal, full_matrices=False)
    dA = a.tangent
    if not compute_uv:
        ds = np.sum(U * (dA @ Vt.T), axis=-2)
        return s, ds

    m, n = a.primal.shape[-2:]
    k = min(m, n)
    ds = np.sum(U * (dA @ Vt.T), axis=-2)

    UtdAV = U.T @ dA @ Vt.T
    F_num = s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2
    mask = np.eye(k, dtype=bool)
    F_num_safe = np.where(mask, np.ones_like(F_num), F_num)
    F = np.where(mask, np.zeros_like(F_num), 1.0 / F_num_safe)

    dU = U @ (F * (UtdAV * s[..., np.newaxis, :] + s[..., :, np.newaxis] * UtdAV.T))
    dVt = (F * (s[..., np.newaxis, :] * UtdAV + UtdAV.T * s[..., :, np.newaxis])) @ Vt

    if full_matrices:
        U_full, _, Vt_full = np.linalg.svd(a.primal, full_matrices=True)
        dU_full = np.zeros_like(U_full)
        dU_full[..., :k] = dU
        dVt_full = np.zeros_like(Vt_full)
        dVt_full[:k, ...] = dVt
        return [
            DualArray(U_full, dU_full),
            DualArray(s, ds),
            DualArray(Vt_full, dVt_full),
        ]

    return [DualArray(U, dU), DualArray(s, ds), DualArray(Vt, dVt)]


@register_func(np.linalg.qr.__name__)
def qr(*args, **kwargs):
    a = args[0]
    mode = kwargs.get("mode", "reduced")
    Q, R = np.linalg.qr(a.primal, mode=mode)
    dA = a.tangent
    m, n = Q.shape
    k = R.shape[0]
    QtdA = Q.T @ dA
    if k == n:
        R_inv = np.linalg.inv(R)
        W = QtdA @ R_inv
        skew = np.tril(W, -1)
        skew = skew - skew.T
        dQ = Q @ skew + (dA - Q @ QtdA) @ R_inv
        dR = QtdA - skew @ R
    else:
        dQ = np.zeros_like(Q)
        dR = QtdA
    return [DualArray(Q, dQ), DualArray(R, dR)]


@register_func(np.linalg.lstsq.__name__)
def lstsq(*args, **kwargs):
    a_arg, b_arg = args[0], args[1]
    rcond = kwargs.get("rcond", None)
    a_primal = (
        a_arg.primal if isinstance(a_arg, DualArray) else np.asarray(a_arg, dtype=float)
    )
    a_tangent = (
        a_arg.tangent if isinstance(a_arg, DualArray) else np.zeros_like(a_primal)
    )
    b_primal = (
        b_arg.primal if isinstance(b_arg, DualArray) else np.asarray(b_arg, dtype=float)
    )
    b_tangent = (
        b_arg.tangent if isinstance(b_arg, DualArray) else np.zeros_like(b_primal)
    )
    result = np.linalg.lstsq(a_primal, b_primal, rcond=rcond)
    x = result[0]
    dx = np.linalg.lstsq(a_primal, b_tangent - a_tangent @ x, rcond=rcond)[0]
    return [DualArray(x, dx), result[1], result[2], result[3]]


@register_func(np.linalg.matrix_power.__name__)
def matrix_power(*args, **kwargs):
    a = args[0]
    n = args[1]
    p = np.linalg.matrix_power(a.primal, n)
    if n == 0:
        return p, np.zeros_like(p)
    if n < 0:
        a_inv = np.linalg.inv(a.primal)
        da_inv = -a_inv @ a.tangent @ a_inv
        result_p = np.linalg.matrix_power(a_inv, -n)
        t = np.zeros_like(result_p)
        for i in range(-n):
            t = t + (
                np.linalg.matrix_power(a_inv, i)
                @ da_inv
                @ np.linalg.matrix_power(a_inv, -n - 1 - i)
            )
        return result_p, t
    t = np.zeros_like(p)
    for i in range(n):
        t = t + (
            np.linalg.matrix_power(a.primal, i)
            @ a.tangent
            @ np.linalg.matrix_power(a.primal, n - 1 - i)
        )
    return p, t


@register_func(np.linalg.multi_dot.__name__)
def multi_dot(*args, **kwargs):
    arrays = args[0]
    primals = []
    tangents = []
    for arr in arrays:
        if isinstance(arr, DualArray):
            primals.append(arr.primal)
            tangents.append(arr.tangent)
        else:
            a = np.asarray(arr, dtype=float)
            primals.append(a)
            tangents.append(np.zeros_like(a))
    p = np.linalg.multi_dot(primals)
    t = np.zeros_like(p)
    for i in range(len(primals)):
        parts = list(primals)
        parts[i] = tangents[i]
        t = t + np.linalg.multi_dot(parts)
    return p, t
