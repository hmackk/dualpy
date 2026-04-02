import numpy as np

from ..core import DualArray, register_func


def _as_dual(a):
    if isinstance(a, DualArray):
        return a
    arr = np.asarray(a, dtype=float)
    return DualArray(arr)


@register_func(np.concatenate.__name__)
def concatenate(*args, **kwargs):
    arrays = [_as_dual(a) for a in args[0]]
    primals = [a.primal for a in arrays]
    tangents = [a.tangent for a in arrays]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.concatenate(primals, **kw), np.concatenate(tangents, **kw)


@register_func(np.stack.__name__)
def stack(*args, **kwargs):
    arrays = [_as_dual(a) for a in args[0]]
    primals = [a.primal for a in arrays]
    tangents = [a.tangent for a in arrays]
    kw = {k: v for k, v in kwargs.items() if k != "out"}
    return np.stack(primals, **kw), np.stack(tangents, **kw)


@register_func(np.reshape.__name__)
def reshape(*args, **kwargs):
    a = args[0]
    new_shape = args[1] if len(args) > 1 else kwargs["shape"]
    return np.reshape(a.primal, new_shape), np.reshape(a.tangent, new_shape)


@register_func(np.transpose.__name__)
def transpose(*args, **kwargs):
    a = args[0]
    axes = args[1] if len(args) > 1 else kwargs.get("axes", None)
    return np.transpose(a.primal, axes), np.transpose(a.tangent, axes)


@register_func(np.swapaxes.__name__)
def swapaxes(*args, **kwargs):
    a = args[0]
    axis1 = args[1]
    axis2 = args[2]
    return np.swapaxes(a.primal, axis1, axis2), np.swapaxes(a.tangent, axis1, axis2)


@register_func(np.moveaxis.__name__)
def moveaxis(*args, **kwargs):
    a = args[0]
    source = args[1]
    destination = args[2]
    return (
        np.moveaxis(a.primal, source, destination),
        np.moveaxis(a.tangent, source, destination),
    )


@register_func(np.squeeze.__name__)
def squeeze(*args, **kwargs):
    a = args[0]
    axis = kwargs.get("axis", None)
    if len(args) > 1:
        axis = args[1]
    return np.squeeze(a.primal, axis=axis), np.squeeze(a.tangent, axis=axis)


@register_func(np.expand_dims.__name__)
def expand_dims(*args, **kwargs):
    a = args[0]
    axis = args[1] if len(args) > 1 else kwargs["axis"]
    return np.expand_dims(a.primal, axis), np.expand_dims(a.tangent, axis)


@register_func(np.ravel.__name__)
def ravel(*args, **kwargs):
    a = args[0]
    return np.ravel(a.primal), np.ravel(a.tangent)


def _stack_variant(np_func):
    def impl(*args, **kwargs):
        tup = [_as_dual(a) for a in args[0]]
        primals = [a.primal for a in tup]
        tangents = [a.tangent for a in tup]
        kw = {k: v for k, v in kwargs.items() if k != "out"}
        return np_func(primals, **kw), np_func(tangents, **kw)

    return impl


for _fn in (np.hstack, np.vstack, np.dstack):
    register_func(_fn.__name__)(_stack_variant(_fn))


@register_func(np.split.__name__)
def split(*args, **kwargs):
    a = args[0]
    indices_or_sections = args[1] if len(args) > 1 else kwargs["indices_or_sections"]
    axis = kwargs.get("axis", 0)
    if len(args) > 2:
        axis = args[2]
    p_parts = np.split(a.primal, indices_or_sections, axis=axis)
    t_parts = np.split(a.tangent, indices_or_sections, axis=axis)
    return [DualArray(p, t) for p, t in zip(p_parts, t_parts, strict=True)]


@register_func(np.array_split.__name__)
def array_split(*args, **kwargs):
    a = args[0]
    indices_or_sections = args[1] if len(args) > 1 else kwargs["indices_or_sections"]
    axis = kwargs.get("axis", 0)
    if len(args) > 2:
        axis = args[2]
    p_parts = np.array_split(a.primal, indices_or_sections, axis=axis)
    t_parts = np.array_split(a.tangent, indices_or_sections, axis=axis)
    return [DualArray(p, t) for p, t in zip(p_parts, t_parts, strict=True)]


@register_func(np.tile.__name__)
def tile(*args, **kwargs):
    a = args[0]
    reps = args[1]
    return np.tile(a.primal, reps), np.tile(a.tangent, reps)


@register_func(np.repeat.__name__)
def repeat(*args, **kwargs):
    a = args[0]
    repeats = args[1] if len(args) > 1 else kwargs["repeats"]
    axis = kwargs.get("axis", None)
    if len(args) > 2:
        axis = args[2]
    return (
        np.repeat(a.primal, repeats, axis=axis),
        np.repeat(a.tangent, repeats, axis=axis),
    )


@register_func(np.flip.__name__)
def flip(*args, **kwargs):
    a = args[0]
    axis = kwargs.get("axis", None)
    if len(args) > 1:
        axis = args[1]
    return np.flip(a.primal, axis=axis), np.flip(a.tangent, axis=axis)


@register_func(np.fliplr.__name__)
def fliplr(*args, **kwargs):
    a = args[0]
    return np.fliplr(a.primal), np.fliplr(a.tangent)


@register_func(np.flipud.__name__)
def flipud(*args, **kwargs):
    a = args[0]
    return np.flipud(a.primal), np.flipud(a.tangent)


@register_func(np.roll.__name__)
def roll(*args, **kwargs):
    a = args[0]
    shift = args[1] if len(args) > 1 else kwargs["shift"]
    axis = kwargs.get("axis", None)
    if len(args) > 2:
        axis = args[2]
    return np.roll(a.primal, shift, axis=axis), np.roll(a.tangent, shift, axis=axis)


@register_func(np.broadcast_to.__name__)
def broadcast_to(*args, **kwargs):
    a = args[0]
    shape = args[1] if len(args) > 1 else kwargs["shape"]
    return np.broadcast_to(a.primal, shape), np.broadcast_to(a.tangent, shape)


@register_func(np.atleast_1d.__name__)
def atleast_1d(*args, **kwargs):
    if len(args) == 1:
        a = _as_dual(args[0])
        return np.atleast_1d(a.primal), np.atleast_1d(a.tangent)
    results = []
    for a in args:
        d = _as_dual(a)
        results.append(DualArray(np.atleast_1d(d.primal), np.atleast_1d(d.tangent)))
    return results


@register_func(np.atleast_2d.__name__)
def atleast_2d(*args, **kwargs):
    if len(args) == 1:
        a = _as_dual(args[0])
        return np.atleast_2d(a.primal), np.atleast_2d(a.tangent)
    results = []
    for a in args:
        d = _as_dual(a)
        results.append(DualArray(np.atleast_2d(d.primal), np.atleast_2d(d.tangent)))
    return results


@register_func(np.atleast_3d.__name__)
def atleast_3d(*args, **kwargs):
    if len(args) == 1:
        a = _as_dual(args[0])
        return np.atleast_3d(a.primal), np.atleast_3d(a.tangent)
    results = []
    for a in args:
        d = _as_dual(a)
        results.append(DualArray(np.atleast_3d(d.primal), np.atleast_3d(d.tangent)))
    return results


@register_func(np.column_stack.__name__)
def column_stack(*args, **kwargs):
    tup = [_as_dual(a) for a in args[0]]
    primals = [a.primal for a in tup]
    tangents = [a.tangent for a in tup]
    return np.column_stack(primals), np.column_stack(tangents)


@register_func(np.pad.__name__)
def pad(*args, **kwargs):
    a = args[0]
    pad_width = args[1] if len(args) > 1 else kwargs["pad_width"]
    mode = kwargs.get("mode", "constant")
    p = np.pad(
        a.primal,
        pad_width,
        mode=mode,
        **{k: v for k, v in kwargs.items() if k not in ("pad_width", "mode")},
    )
    t = np.pad(a.tangent, pad_width, mode="constant", constant_values=0)
    return p, t


@register_func(np.rot90.__name__)
def rot90(*args, **kwargs):
    a = args[0]
    k = args[1] if len(args) > 1 else kwargs.get("k", 1)
    axes = kwargs.get("axes", (0, 1))
    return np.rot90(a.primal, k, axes), np.rot90(a.tangent, k, axes)


@register_func(np.take.__name__)
def take(*args, **kwargs):
    a = args[0]
    indices = args[1] if len(args) > 1 else kwargs["indices"]
    kw = {k: v for k, v in kwargs.items() if k not in ("indices", "out")}
    return np.take(a.primal, indices, **kw), np.take(a.tangent, indices, **kw)


@register_func(np.take_along_axis.__name__)
def take_along_axis(*args, **kwargs):
    a = args[0]
    indices = args[1]
    axis = args[2] if len(args) > 2 else kwargs.get("axis", None)
    return (
        np.take_along_axis(a.primal, indices, axis=axis),
        np.take_along_axis(a.tangent, indices, axis=axis),
    )


@register_func(np.insert.__name__)
def insert(*args, **kwargs):
    a = args[0]
    obj = args[1]
    values = args[2] if len(args) > 2 else kwargs["values"]
    axis = args[3] if len(args) > 3 else kwargs.get("axis", None)
    v_primal = (
        values.primal
        if isinstance(values, DualArray)
        else np.asarray(values, dtype=float)
    )
    v_tangent = (
        values.tangent if isinstance(values, DualArray) else np.zeros_like(v_primal)
    )
    return (
        np.insert(a.primal, obj, v_primal, axis=axis),
        np.insert(a.tangent, obj, v_tangent, axis=axis),
    )


@register_func(np.delete.__name__)
def delete(*args, **kwargs):
    a = args[0]
    obj = args[1]
    axis = args[2] if len(args) > 2 else kwargs.get("axis", None)
    return np.delete(a.primal, obj, axis=axis), np.delete(a.tangent, obj, axis=axis)


@register_func(np.append.__name__)
def append(*args, **kwargs):
    a = args[0]
    values = args[1]
    axis = kwargs.get("axis", None)
    v_primal = (
        values.primal
        if isinstance(values, DualArray)
        else np.asarray(values, dtype=float)
    )
    v_tangent = (
        values.tangent if isinstance(values, DualArray) else np.zeros_like(v_primal)
    )
    return (
        np.append(a.primal, v_primal, axis=axis),
        np.append(a.tangent, v_tangent, axis=axis),
    )


@register_func(np.select.__name__)
def select(*args, **kwargs):
    condlist = args[0]
    choicelist = args[1]
    default = kwargs.get("default", 0)
    conds = []
    for c in condlist:
        if isinstance(c, DualArray):
            conds.append(c.primal.astype(bool))
        else:
            conds.append(np.asarray(c, dtype=bool))
    primals = []
    tangents = []
    for ch in choicelist:
        if isinstance(ch, DualArray):
            primals.append(ch.primal)
            tangents.append(ch.tangent)
        else:
            arr = np.asarray(ch, dtype=float)
            primals.append(arr)
            tangents.append(np.zeros_like(arr))
    d_primal = (
        default.primal
        if isinstance(default, DualArray)
        else np.asarray(default, dtype=float)
    )
    d_tangent = (
        default.tangent
        if isinstance(default, DualArray)
        else np.zeros_like(np.asarray(d_primal, dtype=float))
    )
    return (
        np.select(conds, primals, default=d_primal),
        np.select(conds, tangents, default=d_tangent),
    )
